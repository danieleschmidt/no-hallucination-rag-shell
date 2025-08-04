"""
Command-line interface entry point.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

from .interactive_shell import InteractiveShell


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Configuration file path'
)
@click.option(
    '--threshold', '-t',
    type=float,
    default=0.95,
    help='Factuality threshold (0.0-1.0)'
)
@click.option(
    '--governance-mode', '-g',
    type=click.Choice(['strict', 'balanced', 'permissive']),
    default='strict',
    help='Governance compliance mode'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--log-file',
    type=click.Path(),
    help='Log file path'
)
def main(
    config: Optional[str],
    threshold: float,
    governance_mode: str,
    verbose: bool,
    log_file: Optional[str]
) -> None:
    """
    No-Hallucination RAG Shell - Interactive AI assistant with factuality guarantees.
    
    Examples:
        no-hallucination-shell
        no-hallucination-shell --threshold 0.9 --governance-mode balanced
        no-hallucination-shell --config configs/high_precision.yaml
    """
    # Setup logging
    setup_logging(verbose, log_file)
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        click.echo("Error: Threshold must be between 0.0 and 1.0", err=True)
        sys.exit(1)
    
    # Load configuration
    shell_config = {
        'governance_mode': governance_mode,
        'verbose': verbose
    }
    
    if config:
        # Load config file (simplified for Generation 1)
        click.echo(f"Loading configuration from: {config}")
    
    try:
        # Initialize and run shell
        shell = InteractiveShell(
            config=shell_config,
            factuality_threshold=threshold
        )
        
        shell.run()
        
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def setup_logging(verbose: bool, log_file: Optional[str]) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler (only errors to not interfere with shell)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


if __name__ == '__main__':
    main()