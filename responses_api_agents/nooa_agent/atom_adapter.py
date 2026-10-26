# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Atom sessions through Gym; ported from Paul's gym-atom-bridge at e2a831ef."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from nooa import Agent
from nooa.interactive import Done
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa.unifiedllm import UnifiedLLM
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.items import TaskResult
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.session import Session
from nooa_atom.session.store import SessionHandle, SessionStore, sessions_root

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.task_agent import _latest_user_text


# Root turn methods a Gym rollout may run; children always run handle_batch.
TURN_METHODS = ("handle_batch", "handle_orchestrate")


class GymAtomAgent(Agent):
    """Gym construction shell; the actual task agent is Atom's native session.

    ``turn_method`` comes from the Gym runner's ``nooa.init_kwargs`` and picks
    the root session's turn method.
    """

    def __init__(self, *, llm: UnifiedLLM, turn_method: str = "handle_batch") -> None:
        if turn_method not in TURN_METHODS:
            raise ValueError(f"turn_method must be one of {TURN_METHODS}, got {turn_method!r}")
        super().__init__(llm=llm)
        self.turn_method = turn_method


async def invoke_atom(agent: GymAtomAgent, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    from nooa import tracing

    # These processes belong to one rollout. Keep the complete native trace,
    # including code execution, for export before the sandbox is destroyed.
    trace_dir = Path(mkdtemp(prefix="gym-atom-native-"))
    tracing.enable_tracing(exporters=[tracing.exporters.jsonl(str(trace_dir))])
    gym_hooks = get_hooks()

    class ObservedRegistry(SessionRegistry):
        async def _build(self, options: SessionOptions, handle: SessionHandle, *, restore: bool = False) -> Session:
            session = await super()._build(options, handle, restore=restore)

            def enter() -> None:
                # Atom deliberately starts each session in a fresh Context.
                # Reinstall Gym capture for roots AND delegated children.
                set_hooks(compose_hooks(get_hooks(), gym_hooks))
                tracing.set_session(session.id)
                tracing.register_hooks_in_current_context()

            session.add_loop_context_hook(enter)
            return session

    def same_model(alias: str | None, workspace: Path) -> UnifiedLLM:
        raise ValueError("This Gym rollout permits only its registered policy model")

    options = SessionOptions(
        workspace=Path.cwd(),
        agent_spec=ATOM_AGENT,
        llm=agent.llm,
        turn_method=agent.turn_method,
        sessions_dir=trace_dir / "sessions",
    )
    store = SessionStore(sessions_root(options.workspace, options.sessions_dir))
    registry = ObservedRegistry(store, agent_factory=create_session_agent, llm_factory=same_model)
    try:
        root = await registry.create(options)
        outcome = await root.prompt(_latest_user_text(request))
        if not isinstance(outcome, Done) or outcome.result is None:
            raise RuntimeError(f"Atom ended without a task result: {type(outcome).__name__}")
        return outcome.result
    finally:
        try:
            await registry.close_all()
        finally:
            tracing.shutdown_traces()
