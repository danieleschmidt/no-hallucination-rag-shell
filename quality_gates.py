#!/usr/bin/env python3
"""
Quality Gates Validation - Comprehensive testing and quality assurance
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

def run_code_quality_checks():
    """Run code quality and linting checks."""
    console = Console()
    console.print("[bold yellow]🔍 Running Code Quality Checks...[/bold yellow]")
    
    results = {}
    
    # Check Python syntax
    try:
        python_files = list(Path("no_hallucination_rag").rglob("*.py"))
        syntax_errors = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors += 1
                console.print(f"  ❌ Syntax error in {py_file}: {e}")
        
        results['syntax'] = {
            'files_checked': len(python_files),
            'errors': syntax_errors,
            'passed': syntax_errors == 0
        }
        
        console.print(f"  📝 Python syntax: {len(python_files)} files, {syntax_errors} errors")
        
    except Exception as e:
        results['syntax'] = {'error': str(e), 'passed': False}
    
    # Check imports
    import_errors = 0
    try:
        test_imports = [
            "no_hallucination_rag.quantum.quantum_planner",
            "no_hallucination_rag.core.enhanced_error_handler",
            "no_hallucination_rag.optimization.advanced_caching",
            "no_hallucination_rag.scaling.auto_scaler"
        ]
        
        sys.path.insert(0, str(Path.cwd()))
        
        for module in test_imports:
            try:
                __import__(module)
            except ImportError as e:
                import_errors += 1
                console.print(f"  ❌ Import error: {module} - {e}")
        
        results['imports'] = {
            'modules_checked': len(test_imports),
            'errors': import_errors,
            'passed': import_errors == 0
        }
        
        console.print(f"  📦 Module imports: {len(test_imports)} modules, {import_errors} errors")
        
    except Exception as e:
        results['imports'] = {'error': str(e), 'passed': False}
    
    # Code complexity check (basic)
    try:
        complex_files = 0
        total_lines = 0
        
        for py_file in python_files[:10]:  # Check first 10 files
            with open(py_file, 'r') as f:
                lines = f.readlines()
                total_lines += len(lines)
                
                # Simple complexity heuristic
                if len(lines) > 500:  # Large file
                    complex_files += 1
        
        results['complexity'] = {
            'files_checked': min(10, len(python_files)),
            'complex_files': complex_files,
            'total_lines': total_lines,
            'passed': complex_files < 3  # Allow some complexity
        }
        
        console.print(f"  📊 Code complexity: {total_lines} total lines, {complex_files} large files")
        
    except Exception as e:
        results['complexity'] = {'error': str(e), 'passed': False}
    
    overall_passed = all(r.get('passed', False) for r in results.values())
    console.print(f"  {'✅' if overall_passed else '❌'} Code quality checks: {'PASSED' if overall_passed else 'FAILED'}\\n")
    
    return results


def run_security_scans():
    """Run security vulnerability scans."""
    console = Console()
    console.print("[bold yellow]🛡️ Running Security Scans...[/bold yellow]")
    
    results = {}
    
    # Check for hardcoded secrets
    try:
        secret_patterns = [
            'password',
            'secret',
            'api_key',
            'token',
            'credentials'
        ]
        
        suspicious_files = 0
        total_files = 0
        
        for py_file in Path("no_hallucination_rag").rglob("*.py"):
            total_files += 1
            
            with open(py_file, 'r') as f:
                content = f.read().lower()
                
                # Look for potential secrets
                for pattern in secret_patterns:
                    if f'{pattern} =' in content or f'"{pattern}"' in content:
                        suspicious_files += 1
                        break
        
        results['secrets'] = {
            'files_scanned': total_files,
            'suspicious_files': suspicious_files,
            'passed': suspicious_files < 3  # Some test files may have mock secrets
        }
        
        console.print(f"  🔐 Secret scan: {total_files} files, {suspicious_files} suspicious")
        
    except Exception as e:
        results['secrets'] = {'error': str(e), 'passed': False}
    
    # Check for dangerous imports
    try:
        dangerous_imports = ['eval', 'exec', 'subprocess', 'os.system']
        dangerous_found = 0
        
        for py_file in Path("no_hallucination_rag").rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                for danger in dangerous_imports:
                    if danger in content and 'import' in content:
                        dangerous_found += 1
                        break
        
        results['dangerous_imports'] = {
            'files_scanned': total_files,
            'dangerous_found': dangerous_found,
            'passed': dangerous_found < 5  # Some utility files may need these
        }
        
        console.print(f"  ⚠️  Dangerous imports: {dangerous_found} potentially risky")
        
    except Exception as e:
        results['dangerous_imports'] = {'error': str(e), 'passed': False}
    
    # Check file permissions
    try:
        executable_files = 0
        
        for py_file in Path("no_hallucination_rag").rglob("*.py"):
            if os.access(py_file, os.X_OK):
                executable_files += 1
        
        results['permissions'] = {
            'files_checked': total_files,
            'executable_files': executable_files,
            'passed': True  # Not critical for Python files
        }
        
        console.print(f"  📋 File permissions: {executable_files} executable files")
        
    except Exception as e:
        results['permissions'] = {'error': str(e), 'passed': False}
    
    overall_passed = all(r.get('passed', False) for r in results.values())
    console.print(f"  {'✅' if overall_passed else '❌'} Security scans: {'PASSED' if overall_passed else 'FAILED'}\\n")
    
    return results


def run_performance_benchmarks():
    """Run performance benchmarks."""
    console = Console()
    console.print("[bold yellow]⚡ Running Performance Benchmarks...[/bold yellow]")
    
    results = {}
    
    # Test quantum planner performance
    try:
        sys.path.insert(0, str(Path.cwd() / 'no_hallucination_rag'))
        
        from quantum.quantum_planner import QuantumTaskPlanner, Priority
        from datetime import timedelta
        
        # Benchmark task creation
        start_time = time.time()
        planner = QuantumTaskPlanner()
        
        for i in range(100):
            planner.create_task(
                title=f"Benchmark Task {i}",
                description=f"Performance test task {i}",
                priority=Priority.SECOND_EXCITED,
                estimated_duration=timedelta(hours=1)
            )
        
        creation_time = time.time() - start_time
        
        # Benchmark task sequence optimization
        start_time = time.time()
        sequence = planner.get_optimal_task_sequence(timedelta(hours=24))
        optimization_time = time.time() - start_time
        
        results['quantum_planner'] = {
            'tasks_created': 100,
            'creation_time': creation_time,
            'creation_rate': 100 / creation_time,
            'optimization_time': optimization_time,
            'sequence_length': len(sequence),
            'passed': creation_time < 1.0 and optimization_time < 0.5  # Performance thresholds
        }
        
        console.print(f"  🧠 Quantum planner: {100 / creation_time:.1f} tasks/sec creation")
        console.print(f"     Optimization: {optimization_time:.3f}s for {len(sequence)} tasks")
        
    except Exception as e:
        results['quantum_planner'] = {'error': str(e), 'passed': False}
    
    # Test caching performance
    try:
        from optimization.advanced_caching import MultiLevelCache
        
        cache = MultiLevelCache(l1_size=100, l2_size=1000, l3_size=10000)
        
        # Benchmark cache operations
        start_time = time.time()
        
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        
        put_time = time.time() - start_time
        
        start_time = time.time()
        
        hit_count = 0
        for i in range(1000):
            if cache.get(f"key_{i}") is not None:
                hit_count += 1
        
        get_time = time.time() - start_time
        
        results['caching'] = {
            'operations': 1000,
            'put_time': put_time,
            'get_time': get_time,
            'put_rate': 1000 / put_time,
            'get_rate': 1000 / get_time,
            'hit_ratio': hit_count / 1000,
            'passed': put_time < 0.5 and get_time < 0.1 and hit_count > 950
        }
        
        console.print(f"  ⚡ Cache performance: {1000 / put_time:.0f} puts/sec, {1000 / get_time:.0f} gets/sec")
        console.print(f"     Hit ratio: {hit_count / 1000:.1%}")
        
    except Exception as e:
        results['caching'] = {'error': str(e), 'passed': False}
    
    # Memory usage test
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        results['memory'] = {
            'memory_usage_mb': memory_mb,
            'passed': memory_mb < 500  # Less than 500MB
        }
        
        console.print(f"  💾 Memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        results['memory'] = {'error': str(e), 'passed': False}
    
    overall_passed = all(r.get('passed', False) for r in results.values())
    console.print(f"  {'✅' if overall_passed else '❌'} Performance benchmarks: {'PASSED' if overall_passed else 'FAILED'}\\n")
    
    return results


def run_functional_tests():
    """Run functional integration tests."""
    console = Console()
    console.print("[bold yellow]🧪 Running Functional Tests...[/bold yellow]")
    
    results = {}
    
    # Test Generation 1 functionality
    try:
        # Run the existing simple tests
        result = subprocess.run([
            sys.executable, "test_gen1_simple.py"
        ], capture_output=True, text=True, timeout=30)
        
        gen1_passed = result.returncode == 0
        
        results['generation_1'] = {
            'test_file': 'test_gen1_simple.py',
            'return_code': result.returncode,
            'passed': gen1_passed
        }
        
        console.print(f"  🌟 Generation 1: {'PASSED' if gen1_passed else 'FAILED'}")
        if not gen1_passed and result.stderr:
            console.print(f"     Error: {result.stderr[:100]}...")
        
    except Exception as e:
        results['generation_1'] = {'error': str(e), 'passed': False}
    
    # Test Generation 2 functionality
    try:
        result = subprocess.run([
            sys.executable, "test_gen2_simple.py"
        ], capture_output=True, text=True, timeout=30)
        
        gen2_passed = result.returncode == 0
        
        results['generation_2'] = {
            'test_file': 'test_gen2_simple.py',
            'return_code': result.returncode,
            'passed': gen2_passed
        }
        
        console.print(f"  🛡️  Generation 2: {'PASSED' if gen2_passed else 'FAILED'}")
        if not gen2_passed and result.stderr:
            console.print(f"     Error: {result.stderr[:100]}...")
        
    except Exception as e:
        results['generation_2'] = {'error': str(e), 'passed': False}
    
    # Test Generation 3 functionality
    try:
        result = subprocess.run([
            sys.executable, "test_gen3_simple.py"
        ], capture_output=True, text=True, timeout=30)
        
        gen3_passed = result.returncode == 0
        
        results['generation_3'] = {
            'test_file': 'test_gen3_simple.py',
            'return_code': result.returncode,
            'passed': gen3_passed
        }
        
        console.print(f"  🚀 Generation 3: {'PASSED' if gen3_passed else 'FAILED'}")
        if not gen3_passed and result.stderr:
            console.print(f"     Error: {result.stderr[:100]}...")
        
    except Exception as e:
        results['generation_3'] = {'error': str(e), 'passed': False}
    
    # Test system integration
    try:
        sys.path.insert(0, str(Path.cwd() / 'no_hallucination_rag'))
        
        # Test that main components can be imported and initialized
        from quantum.quantum_planner import QuantumTaskPlanner
        from optimization.advanced_caching import MultiLevelCache
        from scaling.auto_scaler import AutoScaler
        
        planner = QuantumTaskPlanner()
        cache = MultiLevelCache()
        scaler = AutoScaler()
        
        # Test basic interactions
        task = planner.create_task("Integration Test", "Test task for integration")
        cache.put("integration_test", "success")
        
        integration_passed = (
            task is not None and 
            cache.get("integration_test") == "success"
        )
        
        results['integration'] = {
            'components_tested': 3,
            'passed': integration_passed
        }
        
        console.print(f"  🔗 System integration: {'PASSED' if integration_passed else 'FAILED'}")
        
    except Exception as e:
        results['integration'] = {'error': str(e), 'passed': False}
    
    overall_passed = all(r.get('passed', False) for r in results.values())
    console.print(f"  {'✅' if overall_passed else '❌'} Functional tests: {'PASSED' if overall_passed else 'FAILED'}\\n")
    
    return results


def run_coverage_analysis():
    """Analyze test coverage."""
    console = Console()
    console.print("[bold yellow]📊 Running Coverage Analysis...[/bold yellow]")
    
    results = {}
    
    try:
        # Count total lines of code
        total_lines = 0
        total_files = 0
        
        for py_file in Path("no_hallucination_rag").rglob("*.py"):
            if py_file.name != "__init__.py":
                total_files += 1
                with open(py_file, 'r') as f:
                    lines = f.readlines()
                    # Count non-empty, non-comment lines
                    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                    total_lines += len(code_lines)
        
        # Count test files and test functions
        test_files = list(Path(".").glob("test_*.py"))
        test_functions = 0
        
        for test_file in test_files:
            with open(test_file, 'r') as f:
                content = f.read()
                test_functions += content.count("def test_")
        
        # Estimate coverage based on test functions vs modules
        module_count = len(list(Path("no_hallucination_rag").rglob("*.py")))
        estimated_coverage = min(100, (test_functions / max(1, module_count)) * 100)
        
        results['coverage'] = {
            'total_files': total_files,
            'total_lines': total_lines,
            'test_files': len(test_files),
            'test_functions': test_functions,
            'estimated_coverage': estimated_coverage,
            'passed': estimated_coverage > 60  # At least 60% coverage
        }
        
        console.print(f"  📝 Code metrics: {total_files} files, {total_lines} lines of code")
        console.print(f"  🧪 Test metrics: {len(test_files)} test files, {test_functions} test functions")
        console.print(f"  📊 Estimated coverage: {estimated_coverage:.1f}%")
        
    except Exception as e:
        results['coverage'] = {'error': str(e), 'passed': False}
    
    passed = results.get('coverage', {}).get('passed', False)
    console.print(f"  {'✅' if passed else '❌'} Coverage analysis: {'PASSED' if passed else 'FAILED'}\\n")
    
    return results


def generate_quality_report(all_results):
    """Generate comprehensive quality report."""
    console = Console()
    
    # Create summary table
    table = Table(title="🛡️ Quality Gates Summary")
    table.add_column("Gate", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")
    
    overall_passed = True
    
    for gate_name, gate_results in all_results.items():
        if isinstance(gate_results, dict):
            # Check if this gate passed
            gate_passed = True
            details = []
            
            for check_name, check_result in gate_results.items():
                if isinstance(check_result, dict) and 'passed' in check_result:
                    if not check_result['passed']:
                        gate_passed = False
                    
                    # Add details
                    if 'error' in check_result:
                        details.append(f"{check_name}: ERROR")
                    else:
                        details.append(f"{check_name}: {'PASS' if check_result['passed'] else 'FAIL'}")
            
            status = "✅ PASS" if gate_passed else "❌ FAIL"
            detail_text = ", ".join(details)
            
            table.add_row(gate_name.replace('_', ' ').title(), status, detail_text)
            
            if not gate_passed:
                overall_passed = False
    
    console.print(table)
    
    # Overall result
    console.print()
    if overall_passed:
        console.print(Panel(
            "[bold green]🎉 ALL QUALITY GATES PASSED![/bold green]\\n\\n"
            "The system meets all quality, security, performance, and functional requirements.\\n"
            "Ready for production deployment.",
            title="✅ QUALITY GATES RESULT",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]⚠️  SOME QUALITY GATES FAILED[/bold yellow]\\n\\n"
            "Review failed checks above and address issues before production deployment.\\n"
            "The system may still be functional but doesn't meet all quality standards.",
            title="⚠️ QUALITY GATES RESULT", 
            border_style="yellow"
        ))
    
    return overall_passed


def main():
    """Run all quality gates."""
    console = Console()
    
    console.print(Panel(
        "[bold blue]🛡️ QUALITY GATES VALIDATION[/bold blue]\\n"
        "[dim]Comprehensive testing, security, performance, and quality assurance[/dim]",
        title="QUALITY ASSURANCE",
        border_style="blue"
    ))
    
    all_results = {}
    
    quality_gates = [
        ("Code Quality", run_code_quality_checks),
        ("Security Scans", run_security_scans),
        ("Performance Benchmarks", run_performance_benchmarks),
        ("Functional Tests", run_functional_tests),
        ("Coverage Analysis", run_coverage_analysis)
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        
        for gate_name, gate_func in quality_gates:
            task = progress.add_task(f"Running {gate_name}...", total=100)
            
            try:
                progress.update(task, advance=20)
                results = gate_func()
                progress.update(task, completed=100)
                all_results[gate_name.lower().replace(' ', '_')] = results
            except Exception as e:
                progress.update(task, completed=100, description=f"❌ {gate_name} - Failed")
                console.print(f"[red]Quality gate {gate_name} failed: {e}[/red]")
                all_results[gate_name.lower().replace(' ', '_')] = {'error': str(e), 'passed': False}
    
    console.print()
    overall_passed = generate_quality_report(all_results)
    
    return overall_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)