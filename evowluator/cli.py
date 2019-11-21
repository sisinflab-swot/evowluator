import argparse
import os

from . import config
from .config import EXE_NAME
from .data import dataset_converter
from .data.dataset import Dataset
from .data.ontology import Syntax
from .reasoner.base import ReasoningTask
from .evaluation.info import InfoEvaluator
from .evaluation.matchmaking import (
    MatchmakingCorrectnessEvaluator,
    MatchmakingEnergyEvaluator,
    MatchmakingPerformanceEvaluator
)
from .evaluation.ontology import (
    OntologyReasoningCorrectnessEvaluator,
    OntologyReasoningEnergyEvaluator,
    OntologyReasoningPerformanceEvaluator
)
from .evaluation.mode import EvaluationMode
from .visualization.base import Visualizer
from .visualization.plot import LegendLocation


# CLI parser


class SubCommands:
    CLASSIFICATION = 'classification'
    CONSISTENCY = 'consistency'
    MATCHMAKING = 'matchmaking'
    INFO = 'info'
    VISUALIZE = 'visualize'
    CONVERT = 'convert'

    EVALUATION = [CLASSIFICATION, CONSISTENCY, MATCHMAKING]


def process_args() -> int:
    """Runs actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if args.subcommand in SubCommands.EVALUATION:
        if args.num_iterations:
            config.Evaluation.ITERATIONS = args.num_iterations

        if args.timeout:
            config.Evaluation.TIMEOUT = args.timeout

    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI parser."""
    
    # Help parser
    help_parser = argparse.ArgumentParser(add_help=False)

    group = help_parser.add_argument_group('Help and debug')
    group.add_argument('--debug',
                       help='Enable debug output.',
                       action='store_true')
    group.add_argument('-h', '--help',
                       help='Show this help message and exit.',
                       action='help')

    # Mode parser
    mode_parser = argparse.ArgumentParser(add_help=False)
    modes = [m for m in EvaluationMode]

    group = mode_parser.add_argument_group('Mode')
    group.add_argument('-m', '--mode',
                       type=EvaluationMode,
                       choices=modes,
                       default=modes[0],
                       help='Evaluation mode.')
    group.add_argument('-e', '--energy-probe',
                       metavar='CLASS_NAME',
                       help='Probe to use for energy measurements.')

    # Configuration parser
    config_parser = argparse.ArgumentParser(add_help=False)

    group = config_parser.add_argument_group('Configuration')
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

    # Main parser
    main_parser = argparse.ArgumentParser(prog=EXE_NAME,
                                          description='Evaluation framework for OWL reasoners.',
                                          parents=[help_parser],
                                          add_help=False)

    subparsers = main_parser.add_subparsers(title='Available subcommands',
                                            dest='subcommand', required=True)

    # Classification subcommand
    desc = 'Evaluates the classification reasoning task.'
    parser = subparsers.add_parser(SubCommands.CLASSIFICATION,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=classification_sub)

    # Consistency subcommand
    desc = 'Evaluates the consistency reasoning task.'
    parser = subparsers.add_parser(SubCommands.CONSISTENCY,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=consistency_sub)

    # Matchmaking subcommand
    desc = 'Evaluates the matchmaking task.'
    parser = subparsers.add_parser(SubCommands.MATCHMAKING,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=matchmaking_sub)

    # Dataset info subcommand
    desc = 'Prints information about reasoners and datasets.'
    parser = subparsers.add_parser(SubCommands.INFO,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=info_sub)

    # Visualize subcommand
    desc = 'Generates high level statistics and plots.'
    parser = subparsers.add_parser(SubCommands.VISUALIZE,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    parser.add_argument('path', help='Path of the dir containing the results to visualize.')
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

    parser.set_defaults(func=visualize_sub)

    # Convert subcommand
    desc = 'Converts the dataset into the specified syntax.'
    parser = subparsers.add_parser(SubCommands.CONVERT,
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
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

    return main_parser


# Subcommands


def matchmaking_sub(args) -> int:
    evaluator = None

    if args.mode == EvaluationMode.CORRECTNESS:
        evaluator = MatchmakingCorrectnessEvaluator(dataset=args.dataset,
                                                    reasoners=args.reasoners,
                                                    syntax=args.syntax)
    elif args.mode == EvaluationMode.PERFORMANCE:
        evaluator = MatchmakingPerformanceEvaluator(dataset=args.dataset,
                                                    reasoners=args.reasoners,
                                                    syntax=args.syntax)
    elif args.mode == EvaluationMode.ENERGY:
        evaluator = MatchmakingEnergyEvaluator(probe=args.energy_probe,
                                               dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax)
    evaluator.start(args.resume_after)
    return 0


def ontology_reasoning_sub(args, task: ReasoningTask) -> int:
    evaluator = None

    if args.mode == EvaluationMode.CORRECTNESS:
        evaluator = OntologyReasoningCorrectnessEvaluator(task=task,
                                                          dataset=args.dataset,
                                                          reasoners=args.reasoners,
                                                          syntax=args.syntax)
    elif args.mode == EvaluationMode.PERFORMANCE:
        evaluator = OntologyReasoningPerformanceEvaluator(task=task,
                                                          dataset=args.dataset,
                                                          reasoners=args.reasoners,
                                                          syntax=args.syntax)
    elif args.mode == EvaluationMode.ENERGY:
        evaluator = OntologyReasoningEnergyEvaluator(task=task,
                                                     probe=args.energy_probe,
                                                     dataset=args.dataset,
                                                     reasoners=args.reasoners,
                                                     syntax=args.syntax)
    evaluator.start(args.resume_after)
    return 0


def classification_sub(args) -> int:
    return ontology_reasoning_sub(args, ReasoningTask.CLASSIFICATION)


def consistency_sub(args) -> int:
    return ontology_reasoning_sub(args, ReasoningTask.CONSISTENCY)


def info_sub(args) -> int:
    InfoEvaluator(dataset=args.dataset,
                  reasoners=args.reasoners,
                  syntax=args.syntax).start(args.resume_after)
    return 0


def visualize_sub(args) -> int:
    visualizer = Visualizer.from_dir(args.path)

    if args.reasoners:
        visualizer.reasoners = args.reasoners

    if args.size:
        visualizer.figure.size = (args.size[0], args.size[1])

    if args.label_fmt:
        visualizer.figure.label_fmt = args.label_fmt

    visualizer.figure.show_labels = not args.no_labels
    visualizer.figure.show_titles = not args.no_titles
    visualizer.figure.legend_loc = args.legend_loc
    visualizer.figure.legend_cols = args.legend_cols
    visualizer.figure.legend_only = args.legend_only
    visualizer.write_results()

    plots = [p - 1 for p in args.plots] if args.plots else None
    show_gui = not args.no_gui
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
        raise argparse.ArgumentTypeError('{} is not a positive int.'.format(value))
    return ivalue


def positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError('{} is not a positive float.'.format(value))
    return fvalue
