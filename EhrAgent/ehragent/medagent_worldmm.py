from __future__ import annotations

import json
import os
from typing import Optional

from medagent import MedAgent


class MedAgentWorldMM(MedAgent):

    def __init__(
        self,
        *,
        worldmm_llm_name: str,
        worldmm_base_url: str = "",
        worldmm_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._worldmm_llm_name = worldmm_llm_name
        self._worldmm_base_url = (worldmm_base_url or "").rstrip("/")
        self._worldmm_device = worldmm_device
        self._worldmm_timeline_path: Optional[str] = None
        self._ehrmm = None
        self._embed = None

    def set_worldmm_timeline(self, path: Optional[str]) -> None:
        self._worldmm_timeline_path = path

    def set_ehrmm(self, ehrmm) -> None:
        self._ehrmm = ehrmm

    def _ensure_worldmm(self):
        if self._ehrmm is not None:
            return
        from worldmm.memory.ehr import EHRMWorldMemory

        try:
            from worldmm.embedding import EmbeddingModel
            from worldmm.llm import LLMModel

            llm_kw = {}
            if self._worldmm_base_url:
                llm_kw["base_url"] = self._worldmm_base_url
            self._embed = EmbeddingModel(device=self._worldmm_device)
            llm = LLMModel(self._worldmm_llm_name, **llm_kw)
            self._ehrmm = EHRMWorldMemory(embedding_model=self._embed, retriever_llm_model=llm)
        except Exception as exc:
            print(f"[WorldMM] using dependency-light EHR retrieval: {exc}", flush=True)
            self._ehrmm = EHRMWorldMemory(embedding_model=None, retriever_llm_model=None)

    def generate_init_message(self, **context):
        if self.dataset == "mimic_iii":
            from prompts_mimic import EHRAgent_Message_Prompt
        else:
            from prompts_eicu import EHRAgent_Message_Prompt

        self.question = context["message"]
        q = context["message"]

        world_prefix = ""
        p = self._worldmm_timeline_path
        if p and os.path.isfile(p):
            try:
                self._ensure_worldmm()
                assert self._ehrmm is not None
                self._ehrmm.load_mimic_json(p)
                with open(p, encoding="utf-8") as f:
                    until = int(json.load(f).get("until_time", 0))
                ctx = self._ehrmm.memory_context_for_prompt(q, until_time=until)
                world_prefix = (
                    "(WorldMM EHRMWorldMemory: episodic | semantic_kg | belief_memory | self_evolving)\n\n"
                    + ctx
                    + "\n\n---\n\n(EhrAgent LLM medical hints below)\n\n"
                )
            except Exception as e:
                world_prefix = f"(WorldMM EHR memory unavailable: {e})\n\n---\n\n"

        knowledge = self.retrieve_knowledge(self.config_list[0], q)
        self.knowledge = knowledge
        if world_prefix:
            knowledge = world_prefix + knowledge

        examples = self.retrieve_examples(q)
        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=q)
        return init_message

    def execute_function(self, func_call, **kwargs):
        is_exec_success, result = super().execute_function(func_call, **kwargs)
        if self._ehrmm is not None:
            self._ehrmm._current_code = self.code
        return is_exec_success, result
