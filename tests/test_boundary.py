"""Exercise a fresh notebook interpreter with heavyweight imports unavailable."""

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_parent_and_storage_import_without_model_or_grader_dependencies():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','math_verify'}:
            raise ImportError('heavy_dependency_blocked:' + fullname)
        if (fullname.startswith('gstar_opd')
                and not fullname.startswith('gstar_opd_formal')):
            raise ImportError('legacy_import_blocked:' + fullname)
sys.meta_path.insert(0,Block())
from gstar_opd_formal.orchestration.pipeline import Pipeline
from gstar_opd_formal.storage.runs import RunStore
from gstar_opd_formal.storage.restore import restore_training
from gstar_opd_formal.runtime.worker import main
print('boundary_ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_active_python_has_no_legacy_import_edges():
    bad = []
    for path in (ROOT / "src/gstar_opd_formal").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            bad.extend(
                (str(path), name)
                for name in names
                if name.startswith("gstar_opd")
                and not name.startswith("gstar_opd_formal")
            )
    assert bad == []
