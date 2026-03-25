from __future__ import annotations

import re

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType

logger = structlog.get_logger()


class CaptionAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "caption"

    async def run(
        self,
        source_context: str,
        communicative_intent: str,
        final_description: str,
        critique_summary: str,
        diagram_type: DiagramType,
    ) -> str:
        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            prompt_label="caption",
            source_context=source_context,
            communicative_intent=communicative_intent,
            final_description=final_description,
            critique_summary=critique_summary,
        )

        logger.info("Running caption agent", diagram_type=diagram_type.value)

        raw = await self.vlm.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1024,
        )

        cleaned = self._strip_response(raw)
        logger.info("Caption generated", length=len(cleaned))
        return cleaned

    @staticmethod
    def _strip_response(text: str) -> str:
        t = text.strip()
        m = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
        return re.sub(r"\s+", " ", t).strip()
