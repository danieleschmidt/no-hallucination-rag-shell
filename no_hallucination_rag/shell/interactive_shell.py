"""
Interactive shell for no-hallucination RAG system.
"""

import logging
import sys
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

from ..core.factual_rag import FactualRAG, RAGResponse


class InteractiveShell:
    """Interactive command-line interface for RAG system."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        factuality_threshold: float = 0.95
    ):
        self.config = config or {}
        self.factuality_threshold = factuality_threshold
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Initialize RAG system
        self.rag = FactualRAG(
            factuality_threshold=factuality_threshold,
            governance_mode=self.config.get("governance_mode", "strict")
        )
        
        # Shell state
        self.running = True
        self.history: List[Dict[str, Any]] = []
        
    def run(self) -> None:
        """Start the interactive shell."""
        self._print_banner()
        
        try:
            while self.running:
                try:
                    # Get user input
                    query = self._get_user_input()
                    
                    if not query:
                        continue
                    
                    # Handle commands
                    if query.startswith('/'):
                        self._handle_command(query)
                        continue
                    
                    # Process RAG query
                    self._process_query(query)
                    
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'exit' or '/quit' to exit gracefully[/yellow]")
                    continue
                except EOFError:
                    break
                
        except Exception as e:
            self.logger.error(f"Shell error: {e}")
            self.console.print(f"[red]Shell error: {e}[/red]")
        finally:
            self._cleanup()
    
    def _print_banner(self) -> None:
        """Print shell banner."""
        banner = """
🛡️  No-Hallucination RAG Shell v1.0
Type 'help' for commands, 'exit' to quit
"""
        panel = Panel(
            banner.strip(),
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def _get_user_input(self) -> str:
        """Get user input with formatting."""
        factuality_score = self._get_last_factuality_score()
        prompt = f"🛡️  [{factuality_score:.1%}] > "
        
        try:
            return input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            raise
    
    def _get_last_factuality_score(self) -> float:
        """Get factuality score from last query."""
        if self.history:
            return self.history[-1].get("factuality_score", 0.95)
        return 0.95
    
    def _handle_command(self, command: str) -> None:
        """Handle shell commands."""
        cmd_parts = command[1:].split()
        if not cmd_parts:
            return
        
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1:]
        
        if cmd in ['help', 'h']:
            self._show_help()
        elif cmd in ['exit', 'quit', 'q']:
            self.running = False
        elif cmd in ['history', 'hist']:
            self._show_history()
        elif cmd in ['stats', 'statistics']:
            self._show_stats()
        elif cmd in ['config', 'configuration']:
            self._show_config()
        elif cmd == 'clear':
            self.console.clear()
        elif cmd == 'threshold':
            self._set_threshold(args)
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type '/help' for available commands")
    
    def _process_query(self, query: str) -> None:
        """Process RAG query and display results."""
        # Show processing indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("🔍 Retrieving sources...", total=None)
            
            try:
                # Execute query
                response = self.rag.query(query)
                
                progress.update(task, description="✓ Analysis complete")
                
                # Display results
                self._display_response(query, response)
                
                # Store in history
                self.history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": query,
                    "response": response,
                    "factuality_score": response.factuality_score
                })
                
            except Exception as e:
                progress.update(task, description="✗ Error occurred")
                self.console.print(f"[red]Error processing query: {e}[/red]")
                self.logger.error(f"Query processing error: {e}")
    
    def _display_response(self, query: str, response: RAGResponse) -> None:
        """Display RAG response with formatting."""
        self.console.print()
        
        # Status indicators
        status_color = "green" if response.factuality_score >= self.factuality_threshold else "yellow"
        compliance_icon = "✓" if response.governance_compliant else "⚠"
        
        # Response header
        header = f"🔍 Sources: {len(response.sources)} | " \
                f"Factuality: {response.factuality_score:.1%} | " \
                f"Compliance: {compliance_icon}"
        
        self.console.print(f"[{status_color}]{header}[/{status_color}]")
        self.console.print()
        
        # Main answer
        if response.answer:
            answer_panel = Panel(
                response.answer,
                title="Answer",
                border_style=status_color,
                padding=(1, 2)
            )
            self.console.print(answer_panel)
        
        # Sources
        if response.sources:
            self._display_sources(response.sources)
        
        # Citations
        if response.citations:
            self._display_citations(response.citations)
        
        # Warnings or recommendations
        if response.factuality_score < self.factuality_threshold:
            self.console.print(f"[yellow]⚠ Factuality score below threshold ({self.factuality_threshold:.1%})[/yellow]")
        
        if not response.governance_compliant:
            self.console.print("[yellow]⚠ Governance compliance issues detected[/yellow]")
        
        self.console.print()
    
    def _display_sources(self, sources: List[Dict[str, Any]]) -> None:
        """Display source information."""
        if not sources:
            return
        
        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", min_width=30)
        table.add_column("Authority", justify="center", width=10)
        table.add_column("Date", justify="center", width=12)
        
        for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
            title = source.get("title", "Untitled")[:50]
            authority = f"{source.get('authority_score', 0.0):.1%}"
            date = source.get("date", "Unknown")[:10]
            
            table.add_row(str(i), title, authority, date)
        
        self.console.print(table)
        self.console.print()
    
    def _display_citations(self, citations: List[str]) -> None:
        """Display citations."""
        if not citations:
            return
        
        self.console.print("[bold]Sources:[/bold]")
        for citation in citations:
            self.console.print(f"  {citation}")
        self.console.print()
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
**Available Commands:**

• `/help` - Show this help message
• `/exit` - Exit the shell
• `/history` - Show query history
• `/stats` - Show system statistics
• `/config` - Show current configuration
• `/clear` - Clear screen
• `/threshold <value>` - Set factuality threshold (0.0-1.0)

**Usage:**
Simply type your question and press Enter. The system will:
1. Retrieve relevant sources
2. Verify factual accuracy
3. Check governance compliance
4. Provide a response with citations

**Examples:**
• "What are the AI safety requirements?"
• "Explain NIST AI framework guidelines"
• "What are the penalties for AI compliance violations?"
"""
        panel = Panel(
            Markdown(help_text.strip()),
            title="Help",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def _show_history(self) -> None:
        """Show query history."""
        if not self.history:
            self.console.print("[yellow]No queries in history[/yellow]")
            return
        
        table = Table(title="Query History", show_header=True, header_style="bold magenta")
        table.add_column("Time", width=20)
        table.add_column("Query", min_width=40)
        table.add_column("Factuality", justify="center", width=12)
        
        for entry in self.history[-10:]:  # Show last 10 queries
            timestamp = entry["timestamp"][:19].replace("T", " ")
            query = entry["query"][:60] + "..." if len(entry["query"]) > 60 else entry["query"]
            factuality = f"{entry['factuality_score']:.1%}"
            
            table.add_row(timestamp, query, factuality)
        
        self.console.print(table)
    
    def _show_stats(self) -> None:
        """Show system statistics."""
        stats = {
            "Total Queries": len(self.history),
            "Average Factuality": f"{sum(h['factuality_score'] for h in self.history) / max(len(self.history), 1):.1%}",
            "Current Threshold": f"{self.factuality_threshold:.1%}",
            "Knowledge Bases": ", ".join(self.rag.list_knowledge_bases()),
            "Models Loaded": ", ".join(k for k, v in self.rag.models_loaded().items() if v)
        }
        
        table = Table(title="System Statistics", show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        for metric, value in stats.items():
            table.add_row(metric, str(value))
        
        self.console.print(table)
    
    def _show_config(self) -> None:
        """Show current configuration."""
        config_display = {
            "Factuality Threshold": f"{self.factuality_threshold:.1%}",
            "Governance Mode": self.config.get("governance_mode", "strict"),
            "Max Sources": str(self.config.get("max_sources", 10)),
            "Require Citations": str(self.config.get("require_citations", True))
        }
        
        table = Table(title="Configuration", show_header=False)
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        
        for setting, value in config_display.items():
            table.add_row(setting, value)
        
        self.console.print(table)
    
    def _set_threshold(self, args: List[str]) -> None:
        """Set factuality threshold."""
        if not args:
            self.console.print(f"Current threshold: {self.factuality_threshold:.1%}")
            return
        
        try:
            new_threshold = float(args[0])
            if 0.0 <= new_threshold <= 1.0:
                self.factuality_threshold = new_threshold
                self.rag.factuality_threshold = new_threshold
                self.console.print(f"[green]Threshold set to {new_threshold:.1%}[/green]")
            else:
                self.console.print("[red]Threshold must be between 0.0 and 1.0[/red]")
        except ValueError:
            self.console.print("[red]Invalid threshold value[/red]")
    
    def _cleanup(self) -> None:
        """Cleanup on exit."""
        self.console.print("\n[blue]Thanks for using No-Hallucination RAG Shell![/blue]")
        
        if self.history:
            self.console.print(f"[dim]Processed {len(self.history)} queries[/dim]")