# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Atom root/child sessions through Gym without a hosted model."""

import json
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


@pytest.mark.asyncio
async def test_regular_runner_native_root_and_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = [
        "child = await self.delegate('offline child', 'return child evidence')\nreturn_result(Done(explanation='root', result=TaskResult(solution_description='root', evidence=child.result.evidence, how_to_verify='offline', report='native root and child')))",
        "return_result(Done(explanation='child', result=TaskResult(solution_description='child', evidence='native child evidence', how_to_verify='offline', report='child')))",
    ]
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2
        out = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(out, response_id=f"r{len(calls)}"))

    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="responses_api_agents.nooa_agent.atom_adapter:GymAtomAgent",
            invocation_adapter="responses_api_agents.nooa_agent.atom_adapter:invoke_atom",
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=2,
        context_window=1048576,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input="offline native fixture", max_output_tokens=131072
            ),
            model_url_path="/fixture/v1/responses",
            model_cookies={"session": "fixture"},
        )
    )
    assert result.return_value.evidence == "native child evidence"
    assert result.termination_reason is None
    assert len(calls) == len(result.trajectory.turns) == 2
    assert len(result.trajectory.invocations) >= 2
    assert all(call["server_name"] == "policy" and call["url_path"] == "/fixture/v1/responses" for call in calls)
    assert all(call["json"].max_output_tokens == 131072 for call in calls)
    assert all(call["cookies"]["session"] == "fixture" for call in calls)
    print(
        "native result",
        result.return_value.model_dump(),
        "turns",
        len(result.trajectory.turns),
        "invocations",
        len(result.trajectory.invocations),
    )
    (tmp_path / "offline-trajectory.json").write_text(result.trajectory.model_dump_json(indent=2))


ROOT_RESULT = "return_result(Done(explanation='root', result=TaskResult(solution_description='root', evidence='root evidence', how_to_verify='offline', report='root')))"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("init_kwargs", "docstring"),
    [
        ({}, "Work on one task unattended"),
        ({"turn_method": "handle_batch"}, "Work on one task unattended"),
        ({"turn_method": "handle_orchestrate"}, "You work on one task on your own."),
    ],
)
async def test_init_kwargs_choose_the_root_turn_method(tmp_path, monkeypatch, init_kwargs, docstring):
    monkeypatch.chdir(tmp_path)
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        out = NeMoGymResponseFunctionToolCall(
            id="f1", call_id="c1", name="python_cell", arguments=json.dumps({"code": ROOT_RESULT})
        )
        return FakeHTTPResponse(model_response(out, response_id="r1"))

    invocation = NOOAInvocationConfig(
        agent_class="responses_api_agents.nooa_agent.atom_adapter:GymAtomAgent",
        invocation_adapter="responses_api_agents.nooa_agent.atom_adapter:invoke_atom",
        init_kwargs=init_kwargs,
    )
    validate_invocation(invocation)
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=1,
        context_window=1048576,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="offline turn method fixture"),
            model_url_path="/fixture/v1/responses",
            model_cookies={},
        )
    )
    assert result.return_value.evidence == "root evidence"
    assert len(calls) == 1
    # The root's turn-method docstring is the task text the policy sees.
    assert docstring in calls[0]["json"].model_dump_json()


def test_unknown_turn_method_is_rejected():
    from responses_api_agents.nooa_agent.atom_adapter import GymAtomAgent

    with pytest.raises(ValueError, match="turn_method must be one of"):
        GymAtomAgent(llm=None, turn_method="handle")
