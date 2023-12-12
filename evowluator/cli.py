import argparse
import os
from functools import cache

from pyutils.proc.energy import EnergyProbe
from pyutils.types.unit import TimeUnit, MemoryUnit
from .config.debug import Debug, OnError
from .config.evaluation import Evaluation
from .config.paths import EXE_NAME, Paths
from .data import converter
from .data.dataset import Dataset, SortBy, Syntax
from .evaluation import info
from .evaluation.base import CorrectnessEvaluator, PerformanceEvaluator
from .evaluation.mode import EvaluationMode
from .reasoner.base import Reasoner, ReasoningTask
from .util.process import process
from .visualization.base import Visualizer
from .visualization.correctness import CorrectnessStrategy, OracleStrategy
from .visualization.plot import FigureFormat, LegendLocation


# CLI parser


def process_args() -> int:
    args = main_parser().parse_args()

    Debug.TRACE = getattr(args, 'debug', Debug.TRACE)
    Debug.ON_ERROR = getattr(args, 'on_error', Debug.ON_ERROR)
    Evaluation.MODE = getattr(args, 'mode', Evaluation.MODE)
    Evaluation.TIMEOUT = getattr(args, 'timeout', Evaluation.TIMEOUT)
    Evaluation.MAX_WORKERS = getattr(args, 'max_workers', Evaluation.MAX_WORKERS)

    task = getattr(args, 'task', None)
    task = ReasoningTask.with_name(task) if task else ReasoningTask.CLASSIFICATION
    Evaluation.TASK = task

    names = getattr(args, 'reasoners', None)
    Evaluation.REASONERS = Reasoner.with_names(names) if names else Reasoner.supporting_task(task)

    dataset = getattr(args, 'dataset', None)
    dataset = Dataset(dataset) if dataset else Dataset.first()
    dataset.sort_by = getattr(args, 'sort_by', SortBy.NAME)
    dataset.preferred_syntax = getattr(args, 'syntax', None)
    Evaluation.DATASET = dataset

    energy_probes = getattr(args, 'energy_probes', None)
    if energy_probes:
        def get_probe(name: str) -> EnergyProbe:
            probe = EnergyProbe.with_name(name)
            probe.interval = Evaluation.ENERGY_POLLING_INTERVALS.get(probe.name, probe.interval)
            return probe
        Evaluation.ENERGY_PROBES = [get_probe(n) for n in energy_probes]

    if Evaluation.mode() == EvaluationMode.CORRECTNESS:
        Evaluation.ITERATIONS = 1
    else:
        Evaluation.ITERATIONS = getattr(args, 'num_iterations', Evaluation.ITERATIONS)

    strategy = getattr(args, 'correctness_strategy', None)
    if strategy:
        reasoners = [r.name for r in Evaluation.reasoners()]
        Evaluation.CORRECTNESS_STRATEGY = CorrectnessStrategy.with_name(strategy, reasoners)

    results = getattr(args, 'correctness_results', None)
    if results:
        reasoners = [r.name for r in Evaluation.reasoners()]
        Evaluation.CORRECTNESS_STRATEGY = CorrectnessStrategy.with_name(strategy, reasoners)

    return args.func(args)


@cache
def help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Help and debug')
    group.add_argument('--debug',
                       help='Enable debug output.',
                       action='store_true')
    group.add_argument('--on-error',
                       type=OnError,
                       choices=OnError.all(),
                       default=OnError.LOG,
                       help='Error handling strategy.')
    group.add_argument('-h', '--help',
                       help='Show this help message and exit.',
                       action='help')
    return parser


@cache
def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Configuration')
    group.add_argument('-d', '--dataset',
                       help='Desired dataset.')
    group.add_argument('-r', '--reasoners',
                       metavar='REASONER',
                       nargs='+',
                       help='Desired reasoners.')
    group.add_argument('-n', '--num-iterations',
                       type=positive_int,
                       default=Evaluation.iterations(),
                       help='Number of iterations.')
    group.add_argument('-t', '--timeout',
                       default=str(Evaluation.timeout()),
                       help='Timeout in seconds. Can be an arithmetic expression of \"s\", '
                            'the size of the ontology in MB.')
    group.add_argument('-s', '--syntax',
                       type=Syntax,
                       choices=Syntax.all(),
                       help='Use the specified OWL syntax whenever possible.')
    group.add_argument('--sort-by',
                       type=SortBy,
                       choices=SortBy.all(),
                       default=SortBy.NAME,
                       help='Sort the ontologies.')
    return parser


def add_correctness_arguments(parser) -> None:
    strategies = [s.name for s in CorrectnessStrategy.all() if not isinstance(s, OracleStrategy)]
    parser.add_argument('--correctness-strategy',
                        metavar=f'{{{",".join(strategies)},<reasoner>}}',
                        default=strategies[0],
                        help='Strategy or reasoner to use for correctness evaluation.')
    parser.add_argument('--correctness-results',
                        help='Path to correctness results.')


def add_run_parser(subparsers) -> None:
    desc = 'Runs an evaluation.'
    parser = subparsers.add_parser('run',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser(), config_parser()],
                                   add_help=False)
    parser.add_argument('task',
                        choices=[t.name.lower() for t in ReasoningTask.all()],
                        help='Reasoning task to evaluate.')

    modes = EvaluationMode.all()
    parser.add_argument('-m', '--mode',
                        type=EvaluationMode,
                        choices=modes,
                        default=modes[0],
                        help='Evaluation mode.')
    parser.add_argument('-e', '--energy-probes',
                        choices=[p.name.lower() for p in EnergyProbe.all()],
                        nargs='+',
                        help='Probes to use for energy measurements.')
    add_correctness_arguments(parser)
    parser.add_argument('--max-workers',
                        type=positive_int,
                        help='Maximum number of reasoners to run in parallel.')

    parser.set_defaults(func=run_sub)


def add_resume_parser(subparsers) -> None:
    desc = 'Resumes an evaluation.'
    parser = subparsers.add_parser('resume',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)
    parser.add_argument('path',
                        nargs='?',
                        help='Path to the dir containing the evaluation to resume.')
    parser.set_defaults(func=resume_sub)


def add_info_parser(subparsers) -> None:
    desc = 'Prints information about reasoners and datasets.'
    parser = subparsers.add_parser('info',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)
    parser.add_argument('-d', '--dataset',
                        help='Show information about the dataset.')
    parser.add_argument('--json',
                        action='store_true',
                        help='Print information in json format.')
    parser.set_defaults(func=info_sub)


def add_process_parser(subparsers) -> None:
    desc = 'Formats, merges and filters evaluation results.'
    parser = subparsers.add_parser('process',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)
    parser.add_argument('path',
                        nargs='+',
                        help='Paths to the dirs containing the results to merge.')
    parser.add_argument('-d', '--dataset',
                        help='Override the name of the dataset.')
    add_correctness_arguments(parser)
    parser.set_defaults(func=process_sub)


def add_visualize_parser(subparsers) -> None:
    desc = 'Generates high level statistics and plots.'
    parser = subparsers.add_parser('visualize',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)

    parser.add_argument('path',
                        nargs='?',
                        help='Path to the dir containing the results to visualize.')
    strategies = [s.name for s in CorrectnessStrategy.all() if not isinstance(s, OracleStrategy)]
    parser.add_argument('-c', '--correctness-strategy',
                        metavar=f'{{{",".join(strategies)},<reasoner>}}',
                        default=strategies[0],
                        help='Strategy or reasoner to use for correctness evaluation.')
    parser.add_argument('-s', '--size',
                        metavar=('WIDTH', 'HEIGHT'),
                        nargs=2,
                        type=positive_float,
                        help='Width and height of the figure in inches, separated by space.')
    parser.add_argument('-p', '--plots',
                        metavar='PLOT_INDEX',
                        nargs='+',
                        type=positive_int,
                        help='Subplots to show.')
    parser.add_argument('-r', '--reasoners',
                        metavar='REASONER',
                        nargs='+',
                        help='Reasoners to show.')
    parser.add_argument('--exclude-fields',
                        metavar='FIELD',
                        nargs='+',
                        help='Fields to exclude.')
    parser.add_argument('--separate-fields',
                        action='store_true',
                        help='Plot individual fields rather than cumulative metrics.')
    parser.add_argument('--time-unit',
                        choices=TimeUnit.all(),
                        help='Time unit.')
    parser.add_argument('--memory-unit',
                        choices=MemoryUnit.all(),
                        help='Memory unit.')
    parser.add_argument('--no-gui',
                        dest='gui',
                        action='store_false',
                        help='Do not show the interactive GUI.')
    parser.add_argument('--no-titles',
                        dest='show_titles',
                        action='store_false',
                        help='Omit titles for figures and axes.')
    parser.add_argument('--no-labels',
                        dest='show_labels',
                        action='store_false',
                        help='Omit value labels when plotting.')
    parser.add_argument('--figure-fmt',
                        nargs='+',
                        type=FigureFormat,
                        choices=FigureFormat.all(),
                        default=[FigureFormat.PDF],
                        help='File format for the saved figure.')
    parser.add_argument('--label-fmt',
                        help='Float format of value labels.')
    parser.add_argument('--label-rot',
                        type=float,
                        help='Rotation of value labels in degrees.')
    parser.add_argument('--xtick-rot',
                        type=float,
                        help='Rotation of labels on the x axis in degrees.')
    parser.add_argument('--ytick-rot',
                        type=float,
                        help='Rotation of labels on the y axis in degrees.')
    parser.add_argument('--xscale',
                        help='Scale of the x axis.')
    parser.add_argument('--yscale',
                        help='Scale of the y axis.')
    parser.add_argument('--xlimits',
                        metavar=('XMIN', 'XMAX'),
                        nargs=2,
                        type=float,
                        help='Limits of the x axis.')
    parser.add_argument('--ylimits',
                        metavar=('YMIN', 'YMAX'),
                        nargs=2,
                        type=float,
                        help='Limits of the y axis.')
    parser.add_argument('--legend-loc',
                        type=LegendLocation,
                        choices=LegendLocation.all(),
                        default=LegendLocation.BEST,
                        help='Location of the legend.')
    parser.add_argument('--legend-cols',
                        type=positive_int,
                        default=1,
                        help='Number of columns of the legend.')
    parser.add_argument('--legend-only',
                        action='store_true',
                        help='Only plot the legend.')
    parser.add_argument('--colors',
                        metavar='COLOR',
                        nargs='+',
                        help='Plot colors.')
    parser.add_argument('--markers',
                        metavar='MARKER',
                        nargs='+',
                        help='Marker styles.')
    parser.add_argument('--marker-size',
                        type=positive_float,
                        help='Marker size in points.')
    parser.add_argument('--line-styles',
                        metavar='STYLE',
                        nargs='+',
                        help="Line styles.")
    parser.add_argument('--fit-poly',
                        dest='fit_poly_degrees',
                        nargs='+',
                        type=positive_int,
                        default=0,
                        help='Fit a polyline of the specified degree to the data.')
    parser.add_argument('--fit-poly-start-samples',
                        nargs='+',
                        type=non_negative_int,
                        help='Forces the polyline to pass from the average of the first samples.')
    parser.add_argument('--fit-poly-end-samples',
                        nargs='+',
                        type=non_negative_int,
                        help='Forces the polyline to pass from the average of the last samples.')
    parser.add_argument('--edge-alpha',
                        type=float,
                        help='Edge alpha.')
    parser.add_argument('--face-alpha',
                        type=float,
                        help='Face alpha.')
    parser.add_argument('--line-alpha',
                        type=float,
                        help='Line alpha.')
    parser.add_argument('--transparent-bg',
                        action='store_true',
                        help='Use a transparent background when saving the figure.')

    parser.set_defaults(func=visualize_sub)


def add_convert_parser(subparsers) -> None:
    desc = 'Converts the dataset into the specified syntax.'
    parser = subparsers.add_parser('convert',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)

    parser.add_argument('-d', '--dataset',
                        required=True,
                        help='Dataset to convert.')
    parser.add_argument('-s', '--syntax',
                        type=Syntax,
                        choices=Syntax.all(),
                        required=True,
                        help='Desired syntax.')

    parser.set_defaults(func=convert_sub)


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=EXE_NAME,
                                     description='Evaluation framework for OWL reasoners.',
                                     parents=[help_parser()],
                                     add_help=False)
    subparsers = parser.add_subparsers(title='Available subcommands',
                                       dest='subcommand', required=True)
    add_run_parser(subparsers)
    add_resume_parser(subparsers)
    add_info_parser(subparsers)
    add_process_parser(subparsers)
    add_visualize_parser(subparsers)
    add_convert_parser(subparsers)
    return parser


# Subcommands


def run_sub(args) -> int:
    e = CorrectnessEvaluator() if args.mode == EvaluationMode.CORRECTNESS else PerformanceEvaluator()
    e.start(Evaluation.new_config())
    return 0


def resume_sub(args) -> int:
    Evaluation.WORK_DIR = args.path if args.path else Paths.last_results_dir()
    cfg = Evaluation.load_config()
    e = CorrectnessEvaluator() if Evaluation.mode() == EvaluationMode.CORRECTNESS else PerformanceEvaluator()
    e.resume(cfg)
    return 0


def info_sub(args) -> int:
    if args.dataset:
        info.dataset(args.dataset, json_format=args.json)
    else:
        info.general(json_format=args.json)
    return 0


def process_sub(args) -> int:
    process(args.path, correctness_dir=args.correctness_results,
            correctness_strategy=args.correctness_strategy, dataset=args.dataset)
    return 0


def visualize_sub(args) -> int:
    path = args.path if args.path else Paths.last_results_dir()
    visualizer = Visualizer.from_dir(path, reasoners=args.reasoners)

    if hasattr(visualizer, 'set_strategy'):
        visualizer.set_strategy(args.correctness_strategy)

    if hasattr(visualizer, 'fields') and args.exclude_fields:
        for f in args.exclude_fields:
            visualizer.fields.remove(f)

    if hasattr(visualizer, 'separate_fields') and args.separate_fields:
        visualizer.separate_fields = True

    if hasattr(visualizer, 'memory_unit') and args.memory_unit:
        visualizer.memory_unit = args.memory_unit

    if hasattr(visualizer, 'time_unit') and args.time_unit:
        visualizer.time_unit = args.time_unit

    if args.xlimits:
        args.xlimits = (args.xlimits[0], args.xlimits[1])

    if args.ylimits:
        args.ylimits = (args.ylimits[0], args.ylimits[1])

    if args.size:
        args.size = (args.size[0], args.size[1])

    visualizer.figure.set_attrs(**vars(args))
    visualizer.write_results()
    visualizer.plot_results(gui=args.gui, transparent_bg=args.transparent_bg,
                            plots=[p - 1 for p in args.plots] if args.plots else None)

    return 0


def convert_sub(args) -> int:
    dataset_path = os.path.join(Paths.DATA_DIR, args.dataset)
    converter.convert_dataset(Dataset(dataset_path), args.syntax)
    return 0


# Utils


def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f'{value} is a negative int')
    return ivalue

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'{value} is not a positive int')
    return ivalue


def positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError(f'{value} is not a positive float')
    return fvalue
