"""
Command-line interface entry point.
Generation 1: Basic CLI without click dependency.
"""

import logging
import sys
import argparse
from typing import Optional

from .interactive_shell import InteractiveShell


def main():
    """
    No-Hallucination RAG Shell - Interactive AI assistant with factuality guarantees.
    """
    parser = argparse.ArgumentParser(
        description="No-Hallucination RAG Shell - Interactive AI assistant with factuality guarantees"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,  # Lower threshold for Generation 1
        help='Factuality threshold (0.0-1.0, default: 0.85)'
    )
    parser.add_argument(
        '--governance-mode', '-g',
        choices=['strict', 'balanced', 'permissive'],
        default='balanced',  # More permissive for Generation 1
        help='Governance compliance mode (default: balanced)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    shell_config = {
        'governance_mode': args.governance_mode,
        'verbose': args.verbose
    }
    
    if args.config:
        print(f"Loading configuration from: {args.config}")
    
    try:
        # Initialize and run shell
        shell = InteractiveShell(
            config=shell_config,
            factuality_threshold=args.threshold
        )
        
        shell.run()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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