"""Notebook entrypoints keep training and the two evaluation roles separate."""

import json

from gstar_opd_formal.runtime.notebooks import build_notebooks


def test_three_notebooks_call_only_their_selected_pipeline(tmp_path):
    release = {
        "release_id": "a" * 64,
        "drive_path": "GStar_OPD_Formal/releases/example",
        "deployment_sha256": "b" * 64,
        "root_folder_id": "example-drive-root",
    }
    notebooks = build_notebooks(tmp_path, release)
    expected = {
        "Train_Only": "train_pair",
        "Eval_Methods_Only": "evaluate_methods",
        "Eval_Base_Only": "evaluate_base",
    }
    assert len(notebooks) == len(expected)
    for path in notebooks:
        notebook = json.loads(path.read_bytes())
        role = path.stem.removeprefix("GStar_OPD_Formal128_")
        run_cell = "".join(notebook["cells"][-1]["source"])
        assert role in expected
        assert f"result = {expected[role]}(" in run_cell
        assert all(
            f"result = {entry}(" not in run_cell
            for name, entry in expected.items()
            if name != role
        )
        assert ('evaluation_profile="12k_base_l4"' in run_cell) == (
            role == "Eval_Base_Only"
        )
        assert (
            'evaluation_profile=f"12k_a100_b{METHODS_CONCURRENCY}"' in run_cell
        ) == (role == "Eval_Methods_Only")
