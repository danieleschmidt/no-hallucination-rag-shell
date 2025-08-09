#!/usr/bin/env python3
"""
Generation 1 Demo - Showcase working quantum-inspired task planning
"""

import sys
import os
from datetime import timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'no_hallucination_rag'))

from quantum.quantum_planner import QuantumTaskPlanner, Priority, TaskState

def demo_quantum_task_planning():
    """Demonstrate quantum-inspired task planning capabilities."""
    console = Console()
    
    # Welcome message
    console.print(Panel(
        "[bold blue]🌟 QUANTUM-INSPIRED TASK PLANNER DEMO[/bold blue]\n"
        "[dim]Generation 1: Make It Work - Autonomous SDLC Implementation[/dim]",
        title="TERRAGON LABS",
        border_style="blue"
    ))
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner()
    console.print("✨ [green]Quantum Task Planner initialized[/green]")
    console.print()
    
    # Create development tasks
    console.print("[bold yellow]📋 Creating Development Tasks...[/bold yellow]")
    
    tasks_data = [
        ("Implement User Authentication", "JWT-based auth system with refresh tokens", Priority.IONIZED, 6),
        ("Build REST API", "Core API endpoints for user management", Priority.THIRD_EXCITED, 8),
        ("Add Database Layer", "PostgreSQL integration with ORM", Priority.THIRD_EXCITED, 4),
        ("Create Frontend Components", "React components for user interface", Priority.SECOND_EXCITED, 5),
        ("Write Unit Tests", "Comprehensive test coverage", Priority.SECOND_EXCITED, 3),
        ("Setup CI/CD Pipeline", "Automated testing and deployment", Priority.FIRST_EXCITED, 4),
        ("Add Monitoring", "Application performance monitoring", Priority.FIRST_EXCITED, 2),
        ("Update Documentation", "API docs and user guides", Priority.GROUND_STATE, 2),
    ]
    
    created_tasks = []
    for title, desc, priority, hours in tasks_data:
        task = planner.create_task(
            title=title,
            description=desc,
            priority=priority,
            estimated_duration=timedelta(hours=hours)
        )
        created_tasks.append(task)
        console.print(f"  ✅ [cyan]{title}[/cyan] ([dim]{priority.name}, {hours}h[/dim])")
    
    console.print()
    
    # Create quantum entanglements (dependencies)
    console.print("[bold yellow]🔗 Creating Quantum Entanglements...[/bold yellow]")
    entanglements = [
        (0, 1, 0.9),  # Auth -> API
        (1, 2, 0.8),  # API -> Database  
        (1, 3, 0.7),  # API -> Frontend
        (4, 1, 0.8),  # Tests -> API
        (1, 5, 0.6),  # API -> CI/CD
        (5, 6, 0.7),  # CI/CD -> Monitoring
    ]
    
    for i, j, strength in entanglements:
        task1_title = created_tasks[i].title[:20] + "..."
        task2_title = created_tasks[j].title[:20] + "..."
        planner.entangle_tasks(created_tasks[i].id, created_tasks[j].id, strength)
        console.print(f"  🔗 [magenta]{task1_title}[/magenta] ↔ [magenta]{task2_title}[/magenta] ([dim]ρ={strength}[/dim])")
    
    console.print()
    
    # Show quantum state
    quantum_state = planner.get_quantum_state_summary()
    console.print(Panel(
        f"[bold]Total Tasks:[/bold] {quantum_state['total_tasks']}\n"
        f"[bold]Coherent Tasks:[/bold] {quantum_state['coherent_tasks']}\n"
        f"[bold]Entanglement Pairs:[/bold] {quantum_state['entanglement_pairs']}\n"
        f"[bold]Quantum Coherence:[/bold] {quantum_state['quantum_coherence_ratio']:.1%}\n"
        f"[bold]Total Interference:[/bold] {quantum_state['total_quantum_interference']:.3f}",
        title="⚛️  Quantum State Summary",
        border_style="cyan"
    ))
    
    # Get optimal task sequence
    console.print("[bold yellow]🎯 Computing Optimal Task Sequence...[/bold yellow]")
    available_time = timedelta(hours=16)  # 2 work days
    optimal_sequence = planner.get_optimal_task_sequence(available_time)
    
    # Display sequence table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Order", width=6)
    table.add_column("Task", width=35)
    table.add_column("Priority", width=15)  
    table.add_column("Duration", width=10)
    table.add_column("State", width=12)
    
    total_time = timedelta()
    for i, task in enumerate(optimal_sequence, 1):
        total_time += task.estimated_duration
        state_color = {
            TaskState.SUPERPOSITION: "yellow",
            TaskState.COLLAPSED: "green", 
            TaskState.ENTANGLED: "magenta",
            TaskState.COMPLETED: "blue",
            TaskState.FAILED: "red"
        }.get(task.state, "white")
        
        table.add_row(
            str(i),
            task.title[:33] + "..." if len(task.title) > 33 else task.title,
            task.priority.name,
            f"{task.estimated_duration.total_seconds()/3600:.1f}h",
            f"[{state_color}]{task.state.value}[/{state_color}]"
        )
    
    console.print(table)
    console.print(f"📊 [bold]Total Sequence Time:[/bold] {total_time.total_seconds()/3600:.1f}h / {available_time.total_seconds()/3600:.1f}h available")
    console.print()
    
    # Execute task sequence (simulation)
    if optimal_sequence:
        console.print("[bold yellow]⚡ Executing Quantum Task Sequence...[/bold yellow]")
        
        # Execute first few tasks for demonstration
        demo_tasks = optimal_sequence[:3]  # Execute first 3 tasks
        execution_results = planner.execute_task_sequence(demo_tasks)
        
        # Show execution results
        console.print(f"  ✅ [green]Tasks Executed:[/green] {len(execution_results['tasks_executed'])}")
        console.print(f"  ⏱️  [blue]Total Duration:[/blue] {execution_results['total_duration']}")
        console.print(f"  🔬 [purple]Quantum Effects:[/purple] {len(execution_results['quantum_effects_observed'])}")
        
        for effect in execution_results['quantum_effects_observed']:
            console.print(f"     🌊 [dim]{effect['type']}:[/dim] {effect['source_task'][:20]}... → {effect['affected_task'][:20]}...")
    
    console.print()
    
    # Final quantum state
    final_state = planner.get_quantum_state_summary() 
    console.print(Panel(
        f"[green]✨ Quantum Task Planning Demo Complete![/green]\n\n"
        f"[bold]System Status:[/bold]\n"
        f"• [cyan]{final_state['total_tasks']}[/cyan] total tasks managed\n"
        f"• [yellow]{final_state['coherent_tasks']}[/yellow] tasks in superposition\n"
        f"• [magenta]{final_state['entanglement_pairs']}[/magenta] quantum entanglements\n"
        f"• [blue]{final_state['state_distribution'].get('completed', 0)}[/blue] tasks completed\n\n"
        f"[dim]🚀 Generation 1: Basic functionality demonstrated successfully![/dim]",
        title="🎉 DEMO RESULTS",
        border_style="green"
    ))


def main():
    """Run the Generation 1 demonstration."""
    try:
        demo_quantum_task_planning()
        return True
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        return False
    except Exception as e:
        console = Console()
        console.print(f"[red]Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    if success:
        print("✅ GENERATION 1 DEMO SUCCESSFUL")
        print("🚀 Ready to proceed to Generation 2: Make It Robust")
    else:
        print("❌ Generation 1 demo failed")
    exit(0 if success else 1)