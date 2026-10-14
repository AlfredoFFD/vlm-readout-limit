"""Qwen2.5-VL-7B-Instruct 4-bit pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import LocalVLM, load_kwargs_for_dtype


class Qwen25VLClient(LocalVLM):
    provider = "local"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    def _load(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            **load_kwargs_for_dtype(self.dtype),
        )
        self.effective_dtype = self.dtype
        self._model.train(False)

    def _generate(self, messages: list[dict], images: list[Image.Image], max_new_tokens: int) -> str:
        proc = self._processor
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt_text], images=images if images else None,
                      padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        import torch
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _single_turn(self, images, prompt, system, max_new_tokens=8):
        pil_images = [Image.open(p).convert("RGB") for p in images]
        user_content = [{"type": "image"} for _ in pil_images]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages, pil_images, max_new_tokens)

    def _generate_with_scores(self, messages: list[dict], images: list[Image.Image],
                              max_new_tokens: int = 4):
        """Like _generate but also returns the first generated step's logits (1-D)."""
        proc = self._processor
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt_text], images=images if images else None,
                      padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        import torch
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                output_scores=True, return_dict_in_generate=True,
            )
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        text = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        step0_logits = out.scores[0][0].float().cpu().numpy()
        return text, step0_logits

    def two_turn_yes_prob(self, sample_image, sample_prompt, test_image, test_prompt,
                          system, max_new_tokens=4):
        """Two-turn change detection returning (decoded_text, P(YES)) — mirrors
        Qwen3VLClient.two_turn_yes_prob so the 7B logs a graded confidence too."""
        from ..analysis.roc import resolve_yes_no_token_ids, yes_prob_from_scores
        self.ensure_loaded()
        sample = Image.open(sample_image).convert("RGB")
        test = Image.open(test_image).convert("RGB")
        m1 = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sample_prompt},
            ]},
        ]
        ack = self._generate(m1, [sample], max_new_tokens=16) or "READY"
        m2 = m1 + [
            {"role": "assistant", "content": [{"type": "text", "text": ack}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": test_prompt},
            ]},
        ]
        text, step0_logits = self._generate_with_scores(m2, [sample, test],
                                                        max_new_tokens=max_new_tokens)
        tok = getattr(self._processor, "tokenizer", self._processor)
        yes_ids, no_ids = resolve_yes_no_token_ids(tok)
        p_yes = yes_prob_from_scores(step0_logits, yes_ids, no_ids)
        return text, p_yes

    def _two_turn(self, sample_image, sample_prompt, test_image, test_prompt, system, max_new_tokens=8):
        sample = Image.open(sample_image).convert("RGB")
        test = Image.open(test_image).convert("RGB")
        m1 = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sample_prompt},
            ]},
        ]
        ack = self._generate(m1, [sample], max_new_tokens=16) or "READY"
        m2 = m1 + [
            {"role": "assistant", "content": [{"type": "text", "text": ack}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": test_prompt},
            ]},
        ]
        return self._generate(m2, [sample, test], max_new_tokens=max_new_tokens)
