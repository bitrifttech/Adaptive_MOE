"""Main entry point for the Adaptive MoE system.

This module provides a command-line interface for interacting with
the Adaptive MoE system.
"""

import argparse
import sys

from .utils.config import AdaptiveMoEConfig, load_config
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive Mixture of Experts System")

    # Global options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model or expert")
    train_parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )

    # Interactive command
    subparsers.add_parser("interactive", help="Run in interactive mode")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    return parser.parse_args()


def main():
    """Main entry point for the Adaptive MoE system."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override log level from command line
    config.logging.level = args.log_level

    # Set up logging
    setup_logging(config.logging)

    logger.info("Starting Adaptive MoE system")

    # Handle commands
    if args.command == "train":
        train(config, args)
    elif args.command == "serve":
        serve(config, args)
    elif args.command == "interactive":
        interactive(config)
    elif args.command == "version":
        from . import __version__

        print(f"Adaptive MoE v{__version__}")
    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)


def train(config: AdaptiveMoEConfig, args) -> None:
    """Train a new model or expert.

    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    logger.info(f"Arguments: {args}")

    # TODO: Implement training logic
    logger.warning("Training not yet implemented")


def serve(config: AdaptiveMoEConfig, args) -> None:
    """Start the API server.

    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    logger.info(f"Starting API server on {args.host}:{args.port}")

    try:
        # TODO: Implement API server
        logger.warning("API server not yet implemented")

        # Keep the server running
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")


def interactive(config: AdaptiveMoEConfig) -> None:
    """Run in interactive mode.

    Args:
        config: Configuration object.
    """
    logger.info("Starting interactive mode")
    logger.info("Type 'exit' or press Ctrl+C to quit")

    try:
        # TODO: Implement interactive mode
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break

                # TODO: Process user input
                print("Bot: I'm sorry, I'm not yet implemented!")

            except KeyboardInterrupt:
                print()  # New line after ^C
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Exiting interactive mode")


if __name__ == "__main__":
    main()
