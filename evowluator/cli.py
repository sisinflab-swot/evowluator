import argparse
import os

from . import config
from .config import EXE_NAME, Evaluation as EvaluationConfig
from .data import dataset_converter
from .data.dataset import Dataset
from .data.ontology import Ontology
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


# CLI parser


def process_args() -> int:
    """Runs actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if not hasattr(args, 'func'):
        msg = ('Invalid argument(s). Please run "{0} -h" '
               'or "{0} <subcommand> -h" for help.'.format(EXE_NAME))
        raise ValueError(msg)

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
                       help='Probe to use for energy measurements.')

    # Configuration parser
    config_parser = argparse.ArgumentParser(add_help=False)

    group = config_parser.add_argument_group('Configuration')
    group.add_argument('-d', '--dataset',
                       help='Desired dataset.')
    group.add_argument('-r', '--reasoners',
                       nargs='+',
                       help='Desired reasoners.')
    group.add_argument('-n', '--num-iterations',
                       type=positive_int,
                       default=EvaluationConfig.DEFAULT_ITERATIONS,
                       help='Number of iterations.')
    group.add_argument('-s', '--syntax',
                       type=Ontology.Syntax,
                       choices=Ontology.Syntax.all(),
                       help='Use the specified OWL syntax whenever possible.')
    group.add_argument('--resume-after',
                       help='Resume the evaluation after the specified ontology.')

    # Main parser
    main_parser = argparse.ArgumentParser(prog=EXE_NAME,
                                          description='Evaluation framework for OWL reasoners.',
                                          parents=[help_parser],
                                          add_help=False)

    subparsers = main_parser.add_subparsers(title='Available subcommands')

    # Classification subcommand
    desc = 'Evaluates the classification reasoning task.'
    parser = subparsers.add_parser('classification',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=classification_sub)

    # Consistency subcommand
    desc = 'Evaluates the consistency reasoning task.'
    parser = subparsers.add_parser('consistency',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=consistency_sub)

    # Matchmaking subcommand
    desc = 'Evaluates the matchmaking task.'
    parser = subparsers.add_parser('matchmaking',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=matchmaking_sub)

    # Dataset info subcommand
    desc = 'Prints information about reasoners and datasets.'
    parser = subparsers.add_parser('info',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=info_sub)

    # Visualize subcommand
    desc = 'Generates high level statistics and plots.'
    parser = subparsers.add_parser('visualize',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    parser.add_argument('path', help='Path of the dir containing the results to visualize.')
    parser.add_argument('-p', '--plots',
                        nargs='+',
                        type=positive_int,
                        help='Subplots to show.')
    parser.add_argument('-r', '--reasoners',
                        nargs='+',
                        help='Reasoners to show.')

    parser.set_defaults(func=visualize_sub)

    # Convert subcommand
    desc = 'Converts the dataset into the specified syntax.'
    parser = subparsers.add_parser('convert',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)

    parser.add_argument('-d', '--dataset',
                        required=True,
                        help='Dataset to convert.')
    parser.add_argument('-s', '--syntax',
                        type=Ontology.Syntax,
                        choices=Ontology.Syntax.all(),
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
                                                    syntax=args.syntax,
                                                    iterations=args.num_iterations)
    elif args.mode == EvaluationMode.ENERGY:
        evaluator = MatchmakingEnergyEvaluator(probe=args.energy_probe,
                                               dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax,
                                               iterations=args.num_iterations)
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
                                                          syntax=args.syntax,
                                                          iterations=args.num_iterations)
    elif args.mode == EvaluationMode.ENERGY:
        evaluator = OntologyReasoningEnergyEvaluator(task=task,
                                                     probe=args.energy_probe,
                                                     dataset=args.dataset,
                                                     reasoners=args.reasoners,
                                                     syntax=args.syntax,
                                                     iterations=args.num_iterations)
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

    visualizer.write_results()

    plots = [p - 1 for p in args.plots] if args.plots else None
    visualizer.plot_results(plots)
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
