#!/usr/bin/env python3
"""Generate basis path testing report for critical functions."""

import os
import subprocess
from radon.complexity import cc_visit
from coverage import Coverage

# Targeted functions for analysis
TARGET_FUNCTIONS = {
    "classification.py": ["classify_manipulation_advanced"],
    "app2.py": ["ForensicValidator.validate_cross_algorithm"],
    "utils.py": ["delete_selected_history"],
    "main.py": ["main"],
}


def analyze_complexity():
    """Compute cyclomatic complexity for targeted functions."""
    complexity_data = {}
    line_map = {}
    for filename, funcs in TARGET_FUNCTIONS.items():
        if not os.path.exists(filename):
            continue
        with open(filename) as f:
            code = f.read()
        results = cc_visit(code)
        for res in results:
            name = res.fullname
            short = name.split(".")[-1]
            if short in funcs or name in funcs:
                complexity_data[name] = res.complexity
                line_map[name] = (res.lineno, res.endline)
    return complexity_data, line_map


def run_tests_get_coverage():
    """Run pytest under coverage and return coverage object and bug count."""
    cov = Coverage()
    cov.start()
    proc = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    cov.stop()
    cov.save()

    output = proc.stdout + proc.stderr
    bug_count = 0
    if "failed" in output:
        for part in output.split("=="):
            if "failed" in part:
                try:
                    bug_count = int(part.strip().split()[0])
                    break
                except (ValueError, IndexError):
                    pass
    return cov, bug_count


def compute_coverage(cov, line_map):
    """Compute coverage percentage for targeted functions."""
    coverage_info = {}
    for filename, funcs in TARGET_FUNCTIONS.items():
        if not os.path.exists(filename):
            continue
        _, stmts, excl, miss, _ = cov.analysis2(filename)
        executed = set(stmts) - set(miss)
        with open(filename) as f:
            results = cc_visit(f.read())
        for res in results:
            name = res.fullname
            short = name.split(".")[-1]
            if short in funcs or name in funcs:
                start, end = res.lineno, res.endline
                lines = set(range(start, end + 1))
                if not lines:
                    pct = 0.0
                else:
                    pct = len(lines & executed) / len(lines) * 100
                coverage_info[name] = pct
    return coverage_info


def generate_report():
    complexity, line_map = analyze_complexity()
    cov, bug_count = run_tests_get_coverage()
    coverage_data = compute_coverage(cov, line_map)

    report_lines = []
    report_lines.append("# Hasil Basis Path Testing")
    report_lines.append("")
    report_lines.append("| Fungsi | Cyclomatic Complexity | Jalur Dasar | Cakupan Jalur (%) | Bug Ditemukan |")
    report_lines.append("|--------|----------------------|-------------|-------------------|---------------|")

    for func, comp in complexity.items():
        basis_paths = comp
        coverage_pct = coverage_data.get(func, 0.0)
        report_lines.append(
            f"| `{func}` | {comp} | {basis_paths} | {coverage_pct:.1f} | {bug_count} |")

    with open("BASIS_PATH_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))

    print("Report generated: BASIS_PATH_REPORT.md")


if __name__ == "__main__":
    generate_report()
