import ast
import json
import os
import subprocess
from typing import Any, Dict, Tuple

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "pickle",
    "pty",
    "builtins",
}

ALLOWED_OS_CALLS = {"makedirs", "path.join", "path.exists"}


class SafetyError(Exception):
    pass


def validate_code_ast(code: str) -> Tuple[bool, str]:
    """Parses Python code AST to block malicious system or network calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in BLOCKED_MODULES and root_mod != "os":
                    return False, f"Blocked import detected: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in BLOCKED_MODULES:
                    return False, f"Blocked from-import detected: {node.module}"

        # Block eval/exec/open outside outputs
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "eval",
                "exec",
                "__import__",
            }:
                return False, f"Blocked builtin function: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "os" and node.func.attr not in {
                    "makedirs"
                }:
                    return False, f"Blocked os call: os.{node.func.attr}"

    return True, "AST check passed"


def execute_experiment(
    script_path: str, timeout_sec: int = 60
) -> Dict[str, Any]:
    """Runs the candidate script safely after AST validation."""
    with open(script_path, "r") as f:
        code = f.read()

    is_safe, msg = validate_code_ast(code)
    if not is_safe:
        return {
            "success": False,
            "error": f"Security Violation: {msg}",
            "metrics": {},
        }

    try:
        proc = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            return {
                "success": False,
                "error": proc.stderr[-1000:],
                "metrics": {},
            }

        metrics = {}
        if os.path.exists("outputs/metrics.json"):
            with open("outputs/metrics.json", "r") as f:
                metrics = json.load(f)

        return {
            "success": True,
            "stdout": proc.stdout,
            "metrics": metrics,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout_sec}s",
            "metrics": {},
        }
