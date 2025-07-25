from unittest.mock import MagicMock
import os

import huggingface_hub
import pytest

import numpy as np

from trackio import Run, init


class DummyClient:
    def __init__(self):
        self.predict = MagicMock()


def test_run_log_calls_client():
    client = DummyClient()
    run = Run(url="fake_url", project="proj", client=client, name="run1")
    metrics = {"x": 1}
    run.log(metrics)
    client.predict.assert_called_once_with(
        api_name="/log",
        project="proj",
        run="run1",
        metrics=metrics,
        hf_token=huggingface_hub.utils.get_token(),
    )


def test_run_log_image_calls_client():
    client = DummyClient()
    run = Run(url="fake_url", project="proj", client=client, name="run1")
    run.log_image("img.png")
    client.predict.assert_called_with(
        api_name="/log_image",
        project="proj",
        run="run1",
        image_path="img.png",
        hf_token=huggingface_hub.utils.get_token(),
    )


def test_run_log_image_tensor(tmp_path):
    client = DummyClient()
    run = Run(url="fake_url", project="proj", client=client, name="run1")
    arr = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    run.log_image(arr)
    kwargs = client.predict.call_args.kwargs
    assert kwargs["api_name"] == "/log_image"
    assert kwargs["project"] == "proj"
    assert kwargs["run"] == "run1"
    tmp_file = kwargs["image_path"]
    assert tmp_file.endswith(".png")
    assert not os.path.exists(tmp_file)


def test_init_resume_modes(temp_db):
    run = init(
        project="test-project",
        name="new-run",
        resume="never",
    )
    assert isinstance(run, Run)
    assert run.name == "new-run"

    run.log({"x": 1})

    run = init(
        project="test-project",
        name="new-run",
        resume="must",
    )
    assert isinstance(run, Run)
    assert run.name == "new-run"

    run = init(
        project="test-project",
        name="new-run",
        resume="allow",
    )
    assert isinstance(run, Run)
    assert run.name == "new-run"

    run = init(
        project="test-project",
        name="new-run",
        resume="never",
    )
    assert isinstance(run, Run)
    assert run.name != "new-run"

    with pytest.raises(
        ValueError,
        match="Run 'nonexistent-run' does not exist in project 'test-project'",
    ):
        init(
            project="test-project",
            name="nonexistent-run",
            resume="must",
        )

    run = init(
        project="test-project",
        name="nonexistent-run",
        resume="allow",
    )
    assert isinstance(run, Run)
    assert run.name == "nonexistent-run"
