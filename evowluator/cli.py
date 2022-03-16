import argparse
import os
from functools import cache

from pyutils.proc.bench import EnergyProbe
from . import config
from .config import Evaluation, EXE_NAME
from .data import converter
from .data.dataset import Dataset, SortBy, Syntax
from .evaluation import info
from .evaluation.base import CorrectnessEvaluator, PerformanceEvaluator
from .evaluation.mode import EvaluationMode
from .reasoner.base import ReasoningTask
from .util.process import incorrect_ontologies, process
from .visualization.base import Visualizer
from .visualization.correctness import CorrectnessStrategy, OracleStrategy
from .visualization.plot import LegendLocation


# CLI parser


def process_args() -> int:
    args = main_parser().parse_args()

    config.DEBUG = getattr(args, 'debug', config.DEBUG)
    Evaluation.MODE = getattr(args, 'mode', Evaluation.MODE)
    Evaluation.TIMEOUT = getattr(args, 'timeout', Evaluation.TIMEOUT)

    energy_probe = getattr(args, 'energy_probe', None)
    if energy_probe:
        probe = EnergyProbe.with_name(energy_probe)
        probe.interval = Evaluation.ENERGY_POLLING_INTERVALS.get(probe.name, probe.interval)
        Evaluation.ENERGY_PROBE = probe

    if Evaluation.MODE == EvaluationMode.CORRECTNESS:
        Evaluation.ITERATIONS = 1
    else:
        Evaluation.ITERATIONS = getattr(args, 'num_iterations', Evaluation.ITERATIONS)

    return args.func(args)


@cache
def help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Help and debug')
    group.add_argument('--debug',
                       help='Enable debug output.',
                       action='store_true')
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
                       default=Evaluation.ITERATIONS,
                       help='Number of iterations.')
    group.add_argument('-t', '--timeout',
                       type=positive_float,
                       default=Evaluation.TIMEOUT,
                       help='Timeout in seconds.')
    group.add_argument('-s', '--syntax',
                       type=Syntax,
                       choices=Syntax.all(),
                       help='Use the specified OWL syntax whenever possible.')
    group.add_argument('--resume-after',
                       metavar='ONTOLOGY_NAME',
                       help='Resume the evaluation after the specified ontology.')
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
    parser.add_argument('-e', '--energy-probe',
                        choices=[p.name.lower() for p in EnergyProbe.all()],
                        help='Probe to use for energy measurements.')
    add_correctness_arguments(parser)
    parser.add_argument('--max-workers',
                        type=positive_int,
                        help='Maximum number of reasoners to run in parallel.')

    parser.set_defaults(func=run_sub)


def add_info_parser(subparsers) -> None:
    desc = 'Prints information about reasoners and datasets.'
    parser = subparsers.add_parser('info',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)
    parser.add_argument('-d', '--dataset',
                        help='Show information about the dataset.')
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
                        help='Colors to use for each reasoner.')
    parser.add_argument('--markers',
                        metavar='MARKER',
                        nargs='+',
                        help='Markers to use for each reasoner.')
    parser.add_argument('--marker-size',
                        type=positive_float,
                        help='Marker size in points.')
    parser.add_argument('--line-styles',
                        metavar='STYLE',
                        nargs='+',
                        help="Line styles to use for each reasoner.")

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
    add_info_parser(subparsers)
    add_process_parser(subparsers)
    add_visualize_parser(subparsers)
    add_convert_parser(subparsers)
    return parser


# Subcommands


def run_sub(args) -> int:
    evaluator_class = None

    if args.mode == EvaluationMode.CORRECTNESS:
        evaluator_class = CorrectnessEvaluator
    elif args.mode == EvaluationMode.PERFORMANCE:
        evaluator_class = PerformanceEvaluator

    e = evaluator_class(ReasoningTask.with_name(args.task), dataset=args.dataset,
                        reasoners=args.reasoners, syntax=args.syntax)

    if isinstance(e, CorrectnessEvaluator):
        e.set_strategy(args.correctness_strategy)
        e.set_max_workers(args.max_workers)
    elif args.correctness_results:
        for r, o in incorrect_ontologies(args.correctness_results,
                                         args.correctness_strategy).items():
            e.skip_ontologies(r, o)

    e.start(sort_by=args.sort_by, resume_after=args.resume_after)

    return 0


def info_sub(args) -> int:
    if args.dataset:
        info.dataset(args.dataset)
    else:
        info.general()
    return 0


def process_sub(args) -> int:
    process(args.path, correctness_dir=args.correctness_results,
            correctness_strategy=args.correctness_strategy, dataset=args.dataset)
    return 0


def visualize_sub(args) -> int:
    path = args.path if args.path else config.Paths.last_results_dir()
    visualizer = Visualizer.from_dir(path, reasoners=args.reasoners)

    if hasattr(visualizer, 'set_strategy'):
        visualizer.set_strategy(args.correctness_strategy)

    if args.colors:
        visualizer.set_colors(args.colors)

    if args.line_styles:
        visualizer.set_line_styles(args.line_styles)

    if args.markers:
        visualizer.set_markers(args.markers)

    if args.size:
        args.size = (args.size[0], args.size[1])

    visualizer.figure.set_attrs(**vars(args))
    visualizer.write_results()
    visualizer.plot_results(gui=args.gui, plots=[p - 1 for p in args.plots] if args.plots else None)

    return 0


def convert_sub(args) -> int:
    dataset_path = os.path.join(config.Paths.DATA_DIR, args.dataset)
    converter.convert_dataset(Dataset(dataset_path), args.syntax)
    return 0


# Utils


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
