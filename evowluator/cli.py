import argparse
import os

from functools import cache
from pyutils.proc.bench import EnergyProbe

from . import config
from .config import EXE_NAME
from .data import dataset_converter
from .data.dataset import Dataset
from .data.ontology import Syntax
from .evaluation.info import InfoEvaluator
from .evaluation.mode import EvaluationMode
from .evaluation.reasoning import (
    CorrectnessStrategy,
    ReasoningCorrectnessEvaluator,
    ReasoningEnergyEvaluator,
    ReasoningPerformanceEvaluator
)
from .reasoner.base import ReasoningTask
from .visualization.base import Visualizer
from .visualization.plot import LegendLocation


# CLI parser


def process_args() -> int:
    args = main_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if args.subcommand in (t.name for t in ReasoningTask.all()):
        if args.num_iterations:
            config.Evaluation.ITERATIONS = args.num_iterations

        if args.timeout:
            config.Evaluation.TIMEOUT = args.timeout

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
                       default=config.Evaluation.ITERATIONS,
                       help='Number of iterations.')
    group.add_argument('-t', '--timeout',
                       type=positive_float,
                       default=config.Evaluation.TIMEOUT,
                       help='Timeout in seconds.')
    group.add_argument('-s', '--syntax',
                       type=Syntax,
                       choices=Syntax.all(),
                       help='Use the specified OWL syntax whenever possible.')
    group.add_argument('--resume-after',
                       metavar='ONTOLOGY_NAME',
                       help='Resume the evaluation after the specified ontology.')
    return parser


def add_evaluation_parsers(subparsers) -> None:
    mode_parser = argparse.ArgumentParser(add_help=False)
    modes = [m for m in EvaluationMode]

    group = mode_parser.add_argument_group('Mode')
    group.add_argument('-m', '--mode',
                       type=EvaluationMode,
                       choices=modes,
                       default=modes[0],
                       help='Evaluation mode.')
    group.add_argument('-e', '--energy-probe',
                       choices=[p.name.lower() for p in EnergyProbe.all()],
                       help='Probe to use for energy measurements.')
    group.add_argument('-c', '--strategy',
                       choices=[s.name for s in CorrectnessStrategy.all()],
                       default=CorrectnessStrategy.all()[0].name,
                       help='Strategy to use for correctness evaluation.')

    for name in (t.name for t in ReasoningTask.all()):
        desc = f'Evaluates the {name} reasoning task.'
        parser = subparsers.add_parser(name,
                                       description=desc,
                                       help=desc,
                                       parents=[help_parser(), mode_parser, config_parser()],
                                       add_help=False)
        parser.set_defaults(func=globals()[f'{name}_sub'])


def add_info_parser(subparsers) -> None:
    desc = 'Prints information about reasoners and datasets.'
    parser = subparsers.add_parser('info',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser(), config_parser()],
                                   add_help=False)
    parser.set_defaults(func=info_sub)


def add_visualize_parser(subparsers) -> None:
    desc = 'Generates high level statistics and plots.'
    parser = subparsers.add_parser('visualize',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser()],
                                   add_help=False)

    parser.add_argument('path',
                        nargs='?',
                        help='Path of the dir containing the results to visualize.')
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
                        action='store_true',
                        help='Do not show the interactive GUI.')
    parser.add_argument('--no-titles',
                        action='store_true',
                        help='Omit titles for figures and axes.')
    parser.add_argument('--no-labels',
                        action='store_true',
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
    add_evaluation_parsers(subparsers)
    add_info_parser(subparsers)
    add_visualize_parser(subparsers)
    add_convert_parser(subparsers)
    return parser


# Subcommands


def reasoning_sub(args, task: ReasoningTask) -> int:
    evaluator = None

    if args.mode == EvaluationMode.CORRECTNESS:
        evaluator = ReasoningCorrectnessEvaluator(task,
                                                  CorrectnessStrategy.with_name(args.strategy),
                                                  dataset=args.dataset,
                                                  reasoners=args.reasoners,
                                                  syntax=args.syntax)
    elif args.mode == EvaluationMode.PERFORMANCE:
        evaluator = ReasoningPerformanceEvaluator(task,
                                                  dataset=args.dataset,
                                                  reasoners=args.reasoners,
                                                  syntax=args.syntax)
    elif args.mode == EvaluationMode.ENERGY:
        evaluator = ReasoningEnergyEvaluator(task, args.energy_probe,
                                             dataset=args.dataset,
                                             reasoners=args.reasoners,
                                             syntax=args.syntax)
    evaluator.start(args.resume_after)
    return 0


def classification_sub(args) -> int:
    return reasoning_sub(args, ReasoningTask.CLASSIFICATION)


def consistency_sub(args) -> int:
    return reasoning_sub(args, ReasoningTask.CONSISTENCY)


def matchmaking_sub(args) -> int:
    return reasoning_sub(args, ReasoningTask.MATCHMAKING)


def info_sub(args) -> int:
    InfoEvaluator(dataset=args.dataset,
                  reasoners=args.reasoners,
                  syntax=args.syntax).start(args.resume_after)
    return 0


def visualize_sub(args) -> int:
    path = args.path if args.path else config.Paths.last_results_dir()
    visualizer = Visualizer.from_dir(path, reasoners=args.reasoners)
    figure = visualizer.figure

    if args.size:
        figure.size = (args.size[0], args.size[1])

    if args.label_fmt:
        figure.label_fmt = args.label_fmt

    if args.label_rot:
        figure.label_rot = args.label_rot

    if args.xtick_rot:
        figure.xtick_rot = args.xtick_rot

    if args.ytick_rot:
        figure.ytick_rot = args.ytick_rot

    if args.colors:
        visualizer.set_colors(args.colors)

    if args.line_styles:
        visualizer.set_line_styles(args.line_styles)

    if args.markers:
        visualizer.set_markers(args.markers)

    if args.marker_size:
        figure.marker_size = args.marker_size

    figure.show_labels = not args.no_labels
    figure.show_titles = not args.no_titles
    figure.legend_loc = args.legend_loc
    figure.legend_cols = args.legend_cols
    figure.legend_only = args.legend_only

    plots = [p - 1 for p in args.plots] if args.plots else None
    show_gui = not args.no_gui

    visualizer.write_results()
    visualizer.plot_results(gui=show_gui, plots=plots)

    return 0


def convert_sub(args) -> int:
    dataset_path = os.path.join(config.Paths.DATA_DIR, args.dataset)
    dataset_converter.convert(Dataset(dataset_path), args.syntax)
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
