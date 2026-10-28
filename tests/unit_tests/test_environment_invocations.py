# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned model calls keep explicit ownership and role from request to trajectory."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig

import nemo_gym.server_utils
from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.harness_capabilities.checker import inspect_record
from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
from nemo_gym.rollout_collection import _build_trajectory_record, _strip_capture_payloads
from nemo_gym.rollout_correlation import (
    ENVIRONMENT_INVOCATION_HEADER,
    ENVIRONMENT_ROLE_HEADER,
    VERIFIER_INVOCATION_HEADER,
    environment_invocation_context,
    rollout_context,
)
from nemo_gym.server_utils import ServerClient


ROLLOUT = "0-0"


async def test_server_client_declares_environment_owner_only_on_rollout_model_calls(monkeypatch):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=DictConfig(
            {
                "observability_enabled": True,
                "sim": {"responses_api_models": {"m": {"host": "sim-host", "port": 2}}},
                "env": {"environment_servers": {"e": {"host": "env-host", "port": 3}}},
            }
        ),
    )
    request = AsyncMock(return_value="response")
    aiohttp = MagicMock()
    aiohttp.return_value.request = request
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", aiohttp)

    with rollout_context(ROLLOUT):
        await client.post("sim", "/v1/chat/completions", json={})
        with environment_invocation_context("sim-1", role="user_simulator"):
            await client.post("sim", "/v1/chat/completions", json={}, headers={"x-existing": "kept"})
            await client.post("env", "/run", json={})

    plain, owned, other = (call.kwargs for call in request.await_args_list)
    assert ENVIRONMENT_INVOCATION_HEADER not in (plain.get("headers") or {})
    assert owned["url"].endswith(f"/ng-rollout/{ROLLOUT}/v1/chat/completions")
    assert owned["headers"] == {
        "x-existing": "kept",
        ENVIRONMENT_INVOCATION_HEADER: "sim-1",
        ENVIRONMENT_ROLE_HEADER: "user_simulator",
    }
    assert ENVIRONMENT_INVOCATION_HEADER not in (other.get("headers") or {})


async def _capture(store: CaptureStore, *, status: int, response_id: str, headers: list[tuple[str, str]]) -> None:
    async def app(scope, receive, send):
        await receive()
        body = {"id": response_id, "output": []} if status == 200 else {"error": "rejected"}
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": json.dumps(body).encode(), "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b'{"input":"next user turn"}', "more_body": False}

    async def send(_message):
        pass

    await _CaptureMiddleware(app, store=store, model_server_name="aux")(
        {
            "type": "http",
            "path": f"/ng-rollout/{ROLLOUT}/v1/responses",
            "headers": [(key.encode("ascii"), value.encode("ascii")) for key, value in headers],
        },
        receive,
        send,
    )


def _policy_trajectory(policy_call_id: str) -> dict:
    ref = {"model_call_id": policy_call_id}
    return {
        "task_id": "0",
        "rollout_id": ROLLOUT,
        "invocations": [{"invocation_id": "candidate", "status": "completed", "model_calls": [ref]}],
        "turns": [
            {
                "invocation_id": "candidate",
                "task_id": "0",
                "rollout_id": ROLLOUT,
                "turn_no": 1,
                "timestamp": 1.0,
                "step_count": 1,
                "model_calls": [ref],
            }
        ],
    }


async def test_multi_role_capture_keeps_auxiliary_calls_owned_and_out_of_policy_steps(tmp_path):
    store = CaptureStore(tmp_path)
    simulator = [(ENVIRONMENT_INVOCATION_HEADER, "sim-1"), (ENVIRONMENT_ROLE_HEADER, "user_simulator")]
    await _capture(store, status=200, response_id="policy", headers=[])
    await _capture(store, status=503, response_id="", headers=simulator)
    await _capture(store, status=200, response_id="sim", headers=simulator)
    await _capture(store, status=200, response_id="score", headers=[(VERIFIER_INVOCATION_HEADER, "verify-1")])
    policy_call_id = store.read(ROLLOUT)[0]["model_call_id"]

    record = {
        "_ng_rollout_id": ROLLOUT,
        "ng_trajectory": _policy_trajectory(policy_call_id),
        "ng_agent_observations": {"source": "agent", "records": []},
    }
    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)
    trajectory = _build_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, record)

    calls = [call.model_call_id for call in trajectory.model_calls]
    assert len(calls) == 4  # Retries, rejected attempts and auxiliary responses all stay in the evidence.
    [environment] = trajectory.environment_invocations
    assert (environment.invocation_id, environment.role) == ("sim-1", "user_simulator")
    assert [ref.model_call_id for ref in environment.model_calls] == calls[1:3]
    assert [ref.model_call_id for ref in trajectory.verifier_invocations[0].model_calls] == calls[3:]
    assert [turn.model_calls[0].model_call_id for turn in trajectory.turns] == [policy_call_id]

    record["ng_trajectory"] = trajectory.model_dump(mode="json")
    results = {check["id"]: check["status"] for check in inspect_record(record)["checks"]}
    assert results["ownership.call_owner"] == "pass"
    assert results["steps.attempt_accounting"] == "pass"

    # Before the producer declared ownership, the same calls were unowned policy attempts.
    record["ng_trajectory"].pop("environment_invocations")
    results = {check["id"]: check["status"] for check in inspect_record(record)["checks"]}
    assert results["ownership.call_owner"] == "fail"
    assert results["steps.attempt_accounting"] == "fail"


@pytest.mark.parametrize(
    "headers",
    [
        [(ENVIRONMENT_INVOCATION_HEADER, "sim-1")],
        [(ENVIRONMENT_INVOCATION_HEADER, "sim-1"), (ENVIRONMENT_ROLE_HEADER, "unsupported")],
    ],
)
async def test_undeclared_or_unknown_role_keeps_the_call_without_inventing_an_owner(tmp_path, headers):
    store = CaptureStore(tmp_path)
    await _capture(store, status=200, response_id="sim", headers=headers)
    record = {"_ng_rollout_id": ROLLOUT}
    merge_model_call_capture_into_record(record, [tmp_path])

    capture = record["ng_model_call_capture"]
    assert len(capture["calls"]) == 1
    assert "environment_invocations" not in capture
    assert "model_call_capture_records_unreadable" not in {gap["code"] for gap in capture.get("gaps", [])}
    trajectory = _build_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, record)
    record["ng_trajectory"] = trajectory.model_dump(mode="json")
    checks = {check["id"]: check["status"] for check in inspect_record(record)["checks"]}
    assert checks["ownership.call_owner"] == "fail"


def test_producer_trajectory_cannot_declare_environment_ownership():
    record = {
        "ng_trajectory": {
            **_policy_trajectory("policy"),
            "environment_invocations": [
                {"invocation_id": "claimed", "role": "user_simulator", "model_calls": [{"model_call_id": "policy"}]}
            ],
        },
        "ng_model_call_capture": {"calls": [{"model_call_id": "policy", "status_code": 200}]},
    }
    trajectory = _build_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, record)
    assert trajectory.environment_invocations == []


def test_trajectory_omits_empty_environment_invocations():
    trajectory = _build_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, {})
    assert "environment_invocations" not in trajectory.model_dump(mode="json")


async def test_environment_owner_survives_payload_stripping_and_reconstruction(tmp_path):
    store = CaptureStore(tmp_path)
    await _capture(
        store,
        status=200,
        response_id="sim",
        headers=[(ENVIRONMENT_INVOCATION_HEADER, "sim-1"), (ENVIRONMENT_ROLE_HEADER, "user_simulator")],
    )
    record = {"_ng_rollout_id": ROLLOUT}
    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)
    row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
    original = _build_trajectory_record(row, record)
    record["ng_trajectory"] = original.model_dump(mode="json")
    _strip_capture_payloads(record)
    rebuilt = _build_trajectory_record(row, record)
    assert rebuilt.environment_invocations == original.environment_invocations
    assert len(rebuilt.model_calls) == 1


def test_environment_context_rejects_an_unsupported_role():
    with pytest.raises(ValueError, match="unsupported environment role"):
        with environment_invocation_context("sim-1", role="unknown"):
            pytest.fail("Invalid roles must not reach the request")
