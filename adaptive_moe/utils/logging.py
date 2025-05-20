"""Logging configuration for the Adaptive MoE system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    # ANSI color codes
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": WHITE,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    def format(self, record):
        """Format the specified record with colors.

        Args:
            record: LogRecord instance containing all information about the log event.

        Returns:
            Formatted log record with appropriate color codes.
        """
        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = (
                self.COLOR_SEQ % (30 + self.COLORS[levelname])
                + levelname
                + self.RESET_SEQ
            )
            record.levelname = levelname_color

            # Color the message for ERROR and CRITICAL levels
            if levelname in ["ERROR", "CRITICAL"]:
                color_code = self.COLOR_SEQ % (30 + self.COLORS[levelname])
                record.msg = f"{color_code}{record.msg}{self.RESET_SEQ}"

        return super().format(record)


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Set up logging configuration.

    Args:
        config: Logging configuration. If None, default configuration is used.

    Returns:
        Root logger instance.
    """
    if config is None:
        config = LoggingConfig()

    # Create log directory if it doesn't exist
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{config.log_file}"

    # Create console formatter
    color_cyan = ColoredFormatter.COLOR_SEQ % (30 + ColoredFormatter.CYAN)
    color_green = ColoredFormatter.COLOR_SEQ % (30 + ColoredFormatter.GREEN)
    reset = ColoredFormatter.RESET_SEQ

    console_formatter = ColoredFormatter(
        f"{color_cyan}%(asctime)s{reset} - "
        f"{color_green}%(name)s{reset} - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create file formatter
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level)
    console_handler.setFormatter(console_formatter)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.level)
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    logger = logging.getLogger("adaptive_moe")
    logger.setLevel(config.level)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set up third-party loggers
    for logger_name in ["transformers", "datasets", "torch"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Name of the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"adaptive_moe.{name}")
    return logger
