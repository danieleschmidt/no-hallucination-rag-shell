"""
Interactive shell for the No-Hallucination RAG system.
Generation 1: Basic CLI interface with essential functionality.
"""

import logging
import sys
from typing import Dict, Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from ..core.factual_rag import FactualRAG


class InteractiveShell:
    """Interactive command-line interface for RAG queries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, factuality_threshold: float = 0.95):
        self.config = config or {}
        self.factuality_threshold = factuality_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize console
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None
        
        # Initialize RAG system
        try:
            self.rag = FactualRAG(
                factuality_threshold=factuality_threshold,
                governance_mode=self.config.get('governance_mode', 'strict')
            )
            self.logger.info("RAG system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            self.rag = None
        
        self.running = False
        
    def run(self):
        """Start the interactive shell."""
        if not self.rag:
            print("ERROR: RAG system not available. Cannot start shell.")
            return
        
        self.running = True
        self._print_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    if HAS_RICH and self.console:
                        query = self.console.input("[bold cyan]🛡️ >[/] ")
                    else:
                        query = input("🛡️ > ")
                    
                    if not query.strip():
                        continue
                    
                    # Handle commands
                    if query.startswith('/'):
                        self._handle_command(query)
                    else:
                        self._handle_query(query)
                        
                except KeyboardInterrupt:
                    print("\n")
                    break
                except EOFError:
                    print("\nGoodbye!")
                    break
                    
        finally:
            self.running = False
    
    def _print_welcome(self):
        """Print welcome message."""
        welcome_text = """
🛡️ No-Hallucination RAG Shell v1.0 (Generation 1)
Retrieval-First CLI with Zero-Hallucination Guarantees

Type your questions or use commands:
- /help      - Show available commands
- /stats     - Show system statistics  
- /health    - Check system health
- /exit      - Exit the shell

Generation 1 Features:
✅ Basic RAG functionality
✅ Rule-based factuality detection
✅ Simple source ranking
✅ Governance compliance checking
✅ Template-based answer generation

Ask me anything and I'll provide fact-checked answers with source citations!
        """
        
        if HAS_RICH and self.console:
            self.console.print(Panel(welcome_text.strip(), style="cyan"))
        else:
            print(welcome_text)
    
    def _handle_command(self, command: str):
        """Handle shell commands."""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self._show_help()
        elif cmd == '/exit' or cmd == '/quit':
            self.running = False
            print("Goodbye!")
        elif cmd == '/stats':
            self._show_stats()
        elif cmd == '/health':
            self._show_health()
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.")
    
    def _handle_query(self, query: str):
        """Handle user queries."""
        if not self.rag:
            print("ERROR: RAG system not available.")
            return
        
        try:
            # Show processing indicator
            if HAS_RICH and self.console:
                with self.console.status("[bold green]Processing query..."):
                    response = self.rag.query(query)
            else:
                print("🔍 Processing query...")
                response = self.rag.query(query)
            
            # Display response
            self._display_response(response)
            
        except Exception as e:
            print(f"ERROR: Failed to process query: {e}")
            self.logger.error(f"Query processing error: {e}")
    
    def _display_response(self, response):
        """Display RAG response to user."""
        
        if HAS_RICH and self.console:
            # Rich formatted output
            self.console.print(f"\n[bold green]✓[/] Factuality Score: {response.factuality_score:.1%}")
            self.console.print(f"[bold blue]📚[/] Sources: {len(response.sources)}")
            self.console.print(f"[bold yellow]⚖️[/] Governance Compliant: {'Yes' if response.governance_compliant else 'No'}")
            
            self.console.print(Panel(response.answer, title="Answer", style="white"))
            
            if response.citations:
                citations_text = "\n".join(response.citations)
                self.console.print(Panel(citations_text, title="Sources", style="dim"))
                
        else:
            # Plain text output
            print(f"\n✓ Factuality Score: {response.factuality_score:.1%}")
            print(f"📚 Sources: {len(response.sources)}")
            print(f"⚖️ Governance Compliant: {'Yes' if response.governance_compliant else 'No'}")
            print(f"\n--- Answer ---")
            print(response.answer)
            
            if response.citations:
                print(f"\n--- Sources ---")
                for citation in response.citations:
                    print(citation)
        
        print()  # Add spacing
    
    def _show_help(self):
        """Show help information."""
        help_text = """
Available Commands:
  /help      - Show this help message
  /stats     - Show system statistics
  /health    - Check system health
  /exit      - Exit the shell

Usage:
  Simply type your question and press Enter.
  The system will retrieve relevant sources and provide
  a fact-checked answer with citations.

Examples:
  > What is quantum computing?
  > How does AI governance work?
  > What are the latest developments in RAG systems?
        """
        
        if HAS_RICH and self.console:
            self.console.print(Panel(help_text.strip(), title="Help", style="blue"))
        else:
            print(help_text)
    
    def _show_stats(self):
        """Show system statistics."""
        if not self.rag:
            print("ERROR: RAG system not available.")
            return
        
        try:
            health = self.rag.get_system_health()
            performance = self.rag.get_performance_stats()
            
            stats_text = f"""
System Status: {health.get('status', 'unknown')}
Timestamp: {health.get('timestamp', 'unknown')}

Components:
- Retriever: {'✓' if health.get('components', {}).get('retriever') else '✗'}
- Source Ranker: {'✓' if health.get('components', {}).get('source_ranker') else '✗'}
- Factuality Detector: {'✓' if health.get('components', {}).get('factuality_detector') else '✗'}
- Governance Checker: {'✓' if health.get('components', {}).get('governance_checker') else '✗'}

Configuration:
- Factuality Threshold: {self.factuality_threshold:.1%}
- Governance Mode: {self.config.get('governance_mode', 'strict')}
- Generation: 1 (Basic functionality)

Enabled Features:
- Caching: {performance.get('components_enabled', {}).get('caching', False)}
- Metrics: {performance.get('components_enabled', {}).get('metrics', False)}  
- Security: {performance.get('components_enabled', {}).get('security', False)}
- Optimization: {performance.get('components_enabled', {}).get('optimization', False)}
- Concurrency: {performance.get('components_enabled', {}).get('concurrency', False)}
            """
            
            if HAS_RICH and self.console:
                self.console.print(Panel(stats_text.strip(), title="System Statistics", style="green"))
            else:
                print(stats_text)
                
        except Exception as e:
            print(f"ERROR: Failed to get system stats: {e}")
    
    def _show_health(self):
        """Show system health check."""
        if not self.rag:
            print("ERROR: RAG system not available.")
            return
        
        try:
            health = self.rag.get_system_health()
            
            status = health.get('status', 'unknown')
            components = health.get('components', {})
            
            health_text = f"""
Overall Status: {status.upper()}

Component Health:
{'✓ Retriever: OK' if components.get('retriever') else '✗ Retriever: FAILED'}
{'✓ Source Ranker: OK' if components.get('source_ranker') else '✗ Source Ranker: FAILED'}  
{'✓ Factuality Detector: OK' if components.get('factuality_detector') else '✗ Factuality Detector: FAILED'}
{'✓ Governance Checker: OK' if components.get('governance_checker') else '✗ Governance Checker: FAILED'}

System ready for queries: {'Yes' if status == 'healthy' else 'No'}
            """
            
            style = "green" if status == "healthy" else "red"
            
            if HAS_RICH and self.console:
                self.console.print(Panel(health_text.strip(), title="Health Check", style=style))
            else:
                print(health_text)
                
        except Exception as e:
            print(f"ERROR: Failed to get health status: {e}")


def main():
    """Main entry point for the shell."""
    try:
        shell = InteractiveShell()
        shell.run()
    except Exception as e:
        print(f"Failed to start shell: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()