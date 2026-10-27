"""Real checkpoint/Adam/RNG/pending reuse recovery through committed byte objects."""

import time
from types import SimpleNamespace

import pytest
import torch

from gstar_opd_formal.data.dataset import Gsm8kExample
from gstar_opd_formal.methods.u2 import U2Method
from gstar_opd_formal.paths import Layout
from gstar_opd_formal.runtime.session import Session
from gstar_opd_formal.runtime.timing import Timeline
from gstar_opd_formal.storage.files import digest_value, write_json
from gstar_opd_formal.storage.local_recovery import local_training_boundary
from gstar_opd_formal.storage.restore import restore_training
from gstar_opd_formal.storage.runs import open_store
from gstar_opd_formal.training.checkpoint import CheckpointRequest, load_checkpoint
from gstar_opd_formal.training.history import validate_history
from gstar_opd_formal.training.runner import TrainingRun, new_state
from tests.fake_drive import FileDrive
from tests.reuse.test_formal_methods import policy_for


def setup_run(engine, tmp_path):
    binding = {
        "schema": "cpu_test",
        "action": "train",
        "arm": "u2",
        "endpoint": 4,
        "batch_size": 4,
        "updates_per_fresh_batch": 2,
        "definition": {
            "manifests": {"training": {"source_indices": list(range(1024))}}
        },
    }
    drive = FileDrive(tmp_path)
    run_id = digest_value(binding)[:32]
    output = tmp_path / "artifacts/training" / run_id
    output.mkdir(parents=True)
    write_json(output / "binding.json", binding)
    store = open_store(drive, ("runs", run_id, binding, ()))
    store.claim("run")
    study = SimpleNamespace(
        design={"checkpoint_margin_seconds": 0, "checkpoint_interval": 32}
    )
    session = Session(
        study,
        output,
        binding,
        {},
        store,
        Timeline(output, digest_value(binding)),
        time.perf_counter() + 600,
    )
    examples = tuple(Gsm8kExample(index, "1+1?", "#### 2") for index in range(8))
    return TrainingRun(
        session, engine, U2Method(), policy_for(engine), new_state(), examples
    )


def test_real_u2_loop_uses_two_fresh_batches_for_four_updates(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    result = run.loop()
    assert (
        result["completed_updates"],
        result["fresh_batches"],
        result["fresh_question_count"],
        result["horizon_observations"],
    ) == (4, 2, 8, 2)


def test_odd_checkpoint_restores_original_behavior_and_exact_adam(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    run.accept_update()
    saved = run.save("stopped")
    run.accept_update()
    expected = {
        name: tensor.detach().clone()
        for name, tensor in engine.bundle.student.named_parameters()
    }
    payload = torch.load(
        run.session.output / saved["accepted_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    state = load_checkpoint(
        engine,
        run.policy,
        CheckpointRequest(
            run.session.output / saved["accepted_checkpoint"], run.session.binding, {}
        ),
    )
    run.state = state
    run.accept_update()
    assert (
        all(
            torch.equal(value, expected[name])
            for name, value in engine.bundle.student.named_parameters()
        )
        and payload["run_state"]["pending"].accepted_updates == 1
    )


def test_complete_run_restores_all_committed_bytes_without_training(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    result = run.loop()
    run.session.store.release()
    source = run.session.output
    selected = {
        "run_id": source.name,
        "binding": run.session.binding,
        "checkpoint_sha256": result["checkpoint_sha256"],
    }
    source.rename(source.with_name("saved_original"))
    restored = restore_training(
        (
            run.session.store.drive,
            Layout(tmp_path, tmp_path / "artifacts"),
            {"runs_folder_id": "runs"},
        ),
        selected,
    )
    assert (restored["directory"] / result["accepted_checkpoint"]).read_bytes() == (
        source.with_name("saved_original") / result["accepted_checkpoint"]
    ).read_bytes()


def test_corrupted_checkpoint_bytes_are_rejected(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    result = run.loop()
    path = run.session.output / result["accepted_checkpoint"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        local_training_boundary(run.session.output, run.session.binding)


def test_wrong_reuse_index_is_rejected(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    run.accept_update()
    run.state["learner_updates"][0]["reuse_index"] = 1
    with pytest.raises(ValueError):
        validate_history(run.state, run.session.binding, 1)


def test_foreign_writer_blocks_further_commit(engine, tmp_path):
    run = setup_run(engine, tmp_path)
    writer = run.session.store.writer
    value = run.session.store.drive.read_control(writer["file_id"])
    value["writer_token"] = "foreign"
    write_json(writer["path"], value)
    run.session.store.drive.update_control(writer["file_id"], writer["path"])
    with pytest.raises(ValueError):
        run.session.store.require_writer()
