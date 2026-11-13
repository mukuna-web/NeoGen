"""
Code Reader — Read-only introspection of NeoGen simulation modules.

Agents can read module summaries (function signatures, docstrings) to give
code-aware advice. No write access — agents propose changes, humans execute.
"""

import os
import ast
import logging

logger = logging.getLogger(__name__)

NEOGEN_ROOT = os.path.dirname(os.path.dirname(__file__))
SIMULATION_DIR = os.path.join(NEOGEN_ROOT, "simulation")
SWARM_DIR = os.path.join(NEOGEN_ROOT, "swarm")


class CodeReader:
    """Read-only access to NeoGen source code for advisory agents."""

    def __init__(self):
        self._module_cache: dict[str, str] = {}
        self._summary_cache: dict[str, str] = {}

    def read_module_summary(self, module_name: str) -> str:
        """Extract function/class signatures and docstrings from a module.

        This is the cheap option — sends less text to the LLM, saving tokens.
        """
        if module_name in self._summary_cache:
            return self._summary_cache[module_name]

        file_path = self._find_module(module_name)
        if not file_path:
            return f"Module '{module_name}' not found in NeoGen codebase."

        try:
            with open(file_path) as f:
                source = f.read()

            tree = ast.parse(source)
            summary_parts = []

            # Module docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                summary_parts.append(f'"""{docstring[:300]}"""')
                summary_parts.append("")

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ""
                    summary_parts.append(f"class {node.name}:")
                    if class_doc:
                        summary_parts.append(f'    """{class_doc[:200]}"""')

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            sig = self._extract_function_sig(item)
                            summary_parts.append(f"    {sig}")
                            func_doc = ast.get_docstring(item)
                            if func_doc:
                                summary_parts.append(f'        """{func_doc[:150]}"""')
                    summary_parts.append("")

                elif isinstance(node, ast.FunctionDef):
                    sig = self._extract_function_sig(node)
                    summary_parts.append(sig)
                    func_doc = ast.get_docstring(node)
                    if func_doc:
                        summary_parts.append(f'    """{func_doc[:150]}"""')
                    summary_parts.append("")

            result = f"# {module_name} — Summary\n# Path: {file_path}\n\n" + "\n".join(summary_parts)
            self._summary_cache[module_name] = result
            return result

        except Exception as e:
            logger.error("Failed to parse %s: %s", module_name, e)
            return f"Error reading {module_name}: {e}"

    def read_module_full(self, module_name: str, max_lines: int = 200) -> str:
        """Read the full source code of a module (expensive — more LLM tokens).

        Limited to first max_lines to control token cost.
        """
        if module_name in self._module_cache:
            source = self._module_cache[module_name]
        else:
            file_path = self._find_module(module_name)
            if not file_path:
                return f"Module '{module_name}' not found."
            with open(file_path) as f:
                source = f.read()
            self._module_cache[module_name] = source

        lines = source.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n# ... truncated ({len(lines)} total lines)"
        return source

    def list_modules(self) -> list[dict]:
        """List all available NeoGen simulation modules."""
        modules = []
        for directory, label in [(SIMULATION_DIR, "simulation"), (SWARM_DIR, "swarm")]:
            if not os.path.isdir(directory):
                continue
            for fname in sorted(os.listdir(directory)):
                if fname.endswith(".py") and not fname.startswith("_"):
                    fpath = os.path.join(directory, fname)
                    # Get first line of docstring
                    try:
                        with open(fpath) as f:
                            source = f.read()
                        tree = ast.parse(source)
                        doc = ast.get_docstring(tree) or ""
                        first_line = doc.split("\n")[0][:100] if doc else ""
                    except Exception:
                        first_line = ""

                    modules.append({
                        "name": fname,
                        "package": label,
                        "description": first_line,
                        "lines": source.count("\n") + 1 if source else 0,
                    })
        return modules

    def get_module_dependencies(self, module_name: str) -> list[str]:
        """Find which other NeoGen modules this module imports."""
        file_path = self._find_module(module_name)
        if not file_path:
            return []

        try:
            with open(file_path) as f:
                source = f.read()
            tree = ast.parse(source)

            deps = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "neogen" in (node.module or ""):
                        deps.append(node.module)
                    elif node.module and node.level > 0:
                        deps.append(f".{node.module}" if node.module else "(relative)")
            return list(set(deps))
        except Exception:
            return []

    def _find_module(self, module_name: str) -> str | None:
        """Find the file path for a module name."""
        # Add .py if not present
        if not module_name.endswith(".py"):
            module_name += ".py"

        # Search in simulation/ and swarm/
        for directory in [SIMULATION_DIR, SWARM_DIR]:
            path = os.path.join(directory, module_name)
            if os.path.isfile(path):
                return path
        return None

    def _extract_function_sig(self, node: ast.FunctionDef) -> str:
        """Extract a function signature as a string."""
        args = []
        for arg in node.args.args:
            name = arg.arg
            if name == "self":
                args.append("self")
                continue
            annotation = ""
            if arg.annotation:
                try:
                    annotation = f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(f"{name}{annotation}")

        returns = ""
        if node.returns:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        return f"def {node.name}({', '.join(args)}){returns}:"
