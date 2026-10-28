# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from environment_servers.swe_together import model_client
from environment_servers.swe_together.model_client import AuxiliaryModel, error_evidence
from environment_servers.swe_together.simulator import UserAgent
from nemo_gym.rollout_correlation import (
    current_environment_invocation,
    current_verifier_invocation_id,
    verifier_invocation_context,
)


GATEWAY_TIMEOUT = b'{"error":{"message":"litellm.Timeout: Connection timed out. Timeout passed=360.0"}}'


def http_error(status=408, content=GATEWAY_TIMEOUT):
    url = "http://127.0.0.1:1/ng-rollout/one/v1/chat/completions"
    request_info = SimpleNamespace(url=url, real_url=url, method="POST", headers={})
    error = ClientResponseError(request_info=request_info, history=(), status=status, message="Request Timeout")
    error.response_content = content
    return error


async def test_failed_call_keeps_model_server_status_body_and_timing(monkeypatch):
    monkeypatch.setattr(model_client, "raise_for_status", AsyncMock(side_effect=http_error()))
    model = AuxiliaryModel(SimpleNamespace(post=AsyncMock()), "sim", cookies={}, temperature=0.5)
    with pytest.raises(ClientResponseError):
        await model.call("prompt")
    record = model.calls[-1]
    assert record["status_code"] == 408
    assert "Timeout passed=360.0" in record["body"]
    assert record["elapsed_seconds"] >= 0 and record["started_at_unix_seconds"] > 0


async def test_cancelled_call_keeps_its_duration():
    started = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    model = AuxiliaryModel(SimpleNamespace(post=hang), "sim", cookies={}, temperature=0.5)
    task = asyncio.create_task(model.call("prompt"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.calls[-1]["cancelled"] is True and model.calls[-1]["elapsed_seconds"] >= 0


async def test_simulator_wrapper_keeps_the_upstream_cause():
    simulator = UserAgent(SimpleNamespace(call=AsyncMock(side_effect=http_error())))
    with pytest.raises(RuntimeError) as raised:
        await simulator.process("task", "activity", "report", None, 1, True)
    evidence = error_evidence(raised.value)
    assert evidence["message"] == "User simulator request failed"
    assert evidence["cause"].startswith("ClientResponseError: 408")
    assert evidence["status_code"] == 408 and "litellm.Timeout" in evidence["body"]


def test_error_without_http_cause_has_no_status():
    assert error_evidence(ValueError("asset digest mismatch")) == {
        "type": "ValueError",
        "message": "asset digest mismatch",
    }


def _client(owners: list, *, error: Exception | None = None) -> MagicMock:
    async def post(*_args, **_kwargs):
        owners.append((current_environment_invocation(), current_verifier_invocation_id()))
        if error is not None:
            raise error
        return MagicMock(ok=True, read=AsyncMock(return_value=b'{"choices": [{"message": {"content": "next"}}]}'))

    client = MagicMock()
    client.post = post
    return client


async def test_simulator_calls_declare_one_environment_owner_each():
    owners = []
    model = AuxiliaryModel(_client(owners), "sim", cookies={}, temperature=0, environment_role="user_simulator")
    assert (await model.call("first")).content == "next"
    await model.call("second")

    assert [owner for owner, _ in owners] == [
        (record["environment_invocation_id"], "user_simulator") for record in model.calls
    ]
    assert len({record["environment_invocation_id"] for record in model.calls}) == 2
    assert current_environment_invocation() is None


async def test_failed_simulator_call_keeps_its_declared_owner():
    owners = []
    client = _client(owners, error=ConnectionError("reset"))
    model = AuxiliaryModel(client, "sim", cookies={}, temperature=0, environment_role="user_simulator")
    with pytest.raises(ConnectionError):
        await model.call("first")
    assert owners == [((model.calls[0]["environment_invocation_id"], "user_simulator"), None)]
    assert model.calls[0]["error"] == "reset"


async def test_scoring_calls_leave_ownership_to_the_enclosing_verifier():
    owners = []
    model = AuxiliaryModel(_client(owners), "score", cookies={}, temperature=0)
    with verifier_invocation_context("verify-1"):
        await model.call("tag")
    assert owners == [(None, "verify-1")]
    assert "environment_invocation_id" not in model.calls[0]


async def test_prepare_wires_simulator_calls_to_environment_ownership(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from environment_servers.swe_together import app
    from environment_servers.swe_together.tests.test_environment import request, server

    environment = server(tmp_path)
    owners = []
    environment.server_client = _client(owners)
    environment._runtime = SimpleNamespace(prepare=AsyncMock(), manifest={"artifacts": {}})
    req = request()
    episode = app.Episode(req, req.task.task_input.task_data, tmp_path, tmp_path)
    episode.task_metadata = {"record": {"history_policy": "pinned"}, "agent_kwargs": {}}
    episode.dependency_access = SimpleNamespace(register=AsyncMock())
    (tmp_path / "instruction.md").write_text("original")
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="", stderr=""))
    )
    monkeypatch.setattr(app.SWETEnvironmentServer, "_new_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app, "RepositorySnapshots", lambda *a, **k: SimpleNamespace(capture=AsyncMock()))
    monkeypatch.setattr(app, "discover_repo_config_files", AsyncMock(return_value=""))
    monkeypatch.setattr(app, "session_analysis", lambda _: (None, ["original"]))

    await environment._prepare(episode)
    await episode.simulator_model.call("next")
    assert owners == [((episode.simulator_model.calls[0]["environment_invocation_id"], "user_simulator"), None)]
