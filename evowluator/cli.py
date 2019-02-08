import argparse

from . import config
from .config import Test as TestConfig
from .data.ontology import Ontology
from .evaluation.evaluator import Evaluator
from .test.base import NotImplementedTest
from .test.classification import (
    ClassificationCorrectnessTest,
    ClassificationEnergyTest,
    ClassificationMemoryTest,
    ClassificationTimeTest
)
from .test.consistency import (
    ConsistencyCorrectnessTest,
    ConsistencyEnergyTest,
    ConsistencyMemoryTest,
    ConsistencyTimeTest
)
from .test.enum import TestMode
from .test.info import InfoTest
from .test.matchmaking import (MatchmakingEnergyTest, MatchmakingMemoryTest, MatchmakingTimeTest)


# CLI parser


def process_args() -> int:
    """Runs actions based on CLI arguments."""
    args = build_parser().parse_args()

    if args.debug:
        config.DEBUG = True

    if not hasattr(args, 'func'):
        msg = 'Invalid argument(s). Please run "test -h" or "test <subcommand> -h" for help.'
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

    group = mode_parser.add_argument_group('Mode')
    group.add_argument('-m', '--mode',
                       choices=TestMode.ALL,
                       default=TestMode.ALL[0],
                       help='Test mode.')

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
                       default=TestConfig.DEFAULT_ITERATIONS,
                       help='Number of iterations for each test.')
    group.add_argument('-s', '--syntax',
                       choices=Ontology.Syntax.ALL,
                       help='Use the specified OWL syntax whenever possible.')
    group.add_argument('--resume-after',
                       help='Resume the test after the specified ontology.')

    # Main parser
    main_parser = argparse.ArgumentParser(prog='test',
                                          description='Evaluation framework for OWL reasoners.',
                                          parents=[help_parser],
                                          add_help=False)

    subparsers = main_parser.add_subparsers(title='Available test')

    # Classification subcommand
    desc = 'Runs the classification test.'
    parser = subparsers.add_parser('classification',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=classification_sub)

    # Consistency subcommand
    desc = 'Runs the consistency test.'
    parser = subparsers.add_parser('consistency',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser, mode_parser, config_parser],
                                   add_help=False)

    parser.set_defaults(func=consistency_sub)

    # Matchmaking subcommand
    desc = 'Runs the matchmaking test.'
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

    # Evaluate subcommand
    desc = 'Generates evaluation statistics from CSV results.'
    parser = subparsers.add_parser('evaluate',
                                   description=desc,
                                   help=desc,
                                   parents=[help_parser],
                                   add_help=False)
    parser.set_defaults(func=evaluate_sub)
    parser.add_argument('path', help='Path of the results dir to evaluate.')

    return main_parser


# Subcommands


def matchmaking_sub(args) -> int:
    {
        TestMode.CORRECTNESS: NotImplementedTest(),

        TestMode.TIME: MatchmakingTimeTest(dataset=args.dataset,
                                           reasoners=args.reasoners,
                                           syntax=args.syntax,
                                           iterations=args.num_iterations),

        TestMode.MEMORY: MatchmakingMemoryTest(dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax,
                                               iterations=args.num_iterations),

        TestMode.ENERGY: MatchmakingEnergyTest(dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax,
                                               iterations=args.num_iterations)
    }[args.mode].start(args.resume_after)
    return 0


def classification_sub(args) -> int:
    {
        TestMode.CORRECTNESS: ClassificationCorrectnessTest(dataset=args.dataset,
                                                            reasoners=args.reasoners),

        TestMode.TIME: ClassificationTimeTest(dataset=args.dataset,
                                              reasoners=args.reasoners,
                                              syntax=args.syntax,
                                              iterations=args.num_iterations),

        TestMode.MEMORY: ClassificationMemoryTest(dataset=args.dataset,
                                                  reasoners=args.reasoners,
                                                  syntax=args.syntax,
                                                  iterations=args.num_iterations),

        TestMode.ENERGY: ClassificationEnergyTest(dataset=args.dataset,
                                                  reasoners=args.reasoners,
                                                  syntax=args.syntax,
                                                  iterations=args.num_iterations)
    }[args.mode].start(args.resume_after)
    return 0


def consistency_sub(args) -> int:
    {
        TestMode.CORRECTNESS: ConsistencyCorrectnessTest(dataset=args.dataset,
                                                         reasoners=args.reasoners),

        TestMode.TIME: ConsistencyTimeTest(dataset=args.dataset,
                                           reasoners=args.reasoners,
                                           syntax=args.syntax,
                                           iterations=args.num_iterations),

        TestMode.MEMORY: ConsistencyMemoryTest(dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax,
                                               iterations=args.num_iterations),

        TestMode.ENERGY: ConsistencyEnergyTest(dataset=args.dataset,
                                               reasoners=args.reasoners,
                                               syntax=args.syntax,
                                               iterations=args.num_iterations)
    }[args.mode].start(args.resume_after)
    return 0


def info_sub(args) -> int:
    InfoTest(dataset=args.dataset,
             reasoners=args.reasoners).start(args.resume_after)
    return 0


def evaluate_sub(args) -> int:
    evaluator = Evaluator.from_dir(args.path)
    evaluator.write_results()
    evaluator.plot_results()
    return 0


# Utils


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('{} is not a positive int.'.format(value))
    return ivalue
