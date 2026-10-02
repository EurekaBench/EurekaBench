import asyncio
import time

from harbor.agents.installed.claude_code import ClaudeCode as HarborClaudeCode
from harbor.agents.installed.codex import Codex as HarborCodex
from harbor.models.agent.context import AgentContext, ModelUsage

CONTINUE = ("Please continue. These required deliverables do not exist yet: {missing}. "
            "Use the finish tool only after all of them are written.")
ROUNDS = 20
TOTALS = ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")


def add(total, part):
    return part if total is None else total if part is None else total + part


class Continues:
    async def run(self, instruction, environment, context):
        if self._resume:
            return await super().run(instruction, environment, context)
        if len(environment.stages) == 1:
            return await self.work(instruction, environment, context)
        stages = []
        for index, stage in enumerate(environment.stages):
            if index and not await environment.begin_stage(index):
                stages.append({"name": stage["name"], "ran": False})
                continue
            part = AgentContext()
            text = instruction if index == 0 else environment.stage_instruction(index)
            timed_out = False
            try:
                await asyncio.wait_for(self.work(text, environment, part), stage["timeout_sec"])
            except asyncio.TimeoutError:
                timed_out = True
            error = ""
            try:
                self.populate_context_post_run(part)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            environment.end_stage()
            stages.append({"name": stage["name"], "ran": True, "timed_out": timed_out,
                           **{key: getattr(part, key) for key in TOTALS}, **({"error": error} if error else {})})
            for key in TOTALS:
                setattr(context, key, add(getattr(context, key), getattr(part, key)))
            for model, usage in (part.model_usage or {}).items():
                context.model_usage = context.model_usage or {}
                total = context.model_usage.setdefault(model, ModelUsage())
                for key in TOTALS:
                    setattr(total, key, add(getattr(total, key), getattr(usage, key)))
        context.metadata = {"stages": stages}

    async def work(self, instruction, environment, context):
        started = time.monotonic()
        await super().run(instruction, environment, context)
        rounds = 0
        while rounds < environment.metadata.get("continue_rounds", ROUNDS):
            missing = environment.missing_deliverables()
            if not missing or time.monotonic() - started > 0.95 * environment.stage_timeout_sec:
                return
            round_started, stops = time.monotonic(), environment.search_stops()
            await self.resume(CONTINUE.format(missing=", ".join(missing)), environment, context)
            if environment.search_stops() == stops:
                rounds += 1
            if time.monotonic() - round_started < 30:
                await asyncio.sleep(60)


class Codex(Continues, HarborCodex):
    pass


class ClaudeCode(Continues, HarborClaudeCode):
    pass
