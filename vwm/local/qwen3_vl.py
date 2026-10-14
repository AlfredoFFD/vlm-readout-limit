"""Qwen3-VL-4B-Instruct 4-bit pipeline (transformers >= 4.57).

Hugging Face IDs:
- "Qwen/Qwen3-VL-4B-Instruct"

Notes:
- The processor uses the standard chat template; we pass content blocks as
  [{"type": "image", "image": <PIL.Image>}, {"type": "text", "text": ...}].
- For two-turn (C2), we keep the prior user image + assistant ack in the messages list
  so the model attends to both images via cross-attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import LocalVLM, load_kwargs_for_dtype


class Qwen3VLClient(LocalVLM):
    provider = "local"
    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    def _load(self) -> None:
        # transformers >= 5.9 ships Qwen3VLForConditionalGeneration directly.
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            **load_kwargs_for_dtype(self.dtype),
        )
        self.effective_dtype = self.dtype
        self._model.train(False)   # inference mode

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        proc = self._processor
        # Extract images from messages for the vision encoder.
        images: list[Image.Image] = []
        for m in messages:
            for blk in m.get("content", []):
                if isinstance(blk, dict) and blk.get("type") == "image":
                    images.append(blk["image"])
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt_text], images=images if images else None,
                      return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        import torch
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _single_turn(
        self,
        images: Sequence[Path],
        prompt: str,
        system: str,
        max_new_tokens: int = 8,
    ) -> str:
        pil_images = [Image.open(p).convert("RGB") for p in images]
        user_content = [{"type": "image", "image": im} for im in pil_images]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]
        return self._generate(messages, max_new_tokens)

    def _two_turn(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_new_tokens: int = 8,
    ) -> str:
        sample = Image.open(sample_image).convert("RGB")
        test = Image.open(test_image).convert("RGB")
        m1 = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample},
                {"type": "text", "text": sample_prompt},
            ]},
        ]
        ack = self._generate(m1, max_new_tokens=16) or "READY"
        m2 = m1 + [
            {"role": "assistant", "content": [{"type": "text", "text": ack}]},
            {"role": "user", "content": [
                {"type": "image", "image": test},
                {"type": "text", "text": test_prompt},
            ]},
        ]
        return self._generate(m2, max_new_tokens=max_new_tokens)

    # ----- M4: continuous decision signal (SDT ROC) -----
    # GPU-only: validate on a Colab A100 next session (local GPU is down). This is a
    # thin seam over the CPU-tested primitives in vwm.analysis.roc; the YES/NO logit
    # extraction math and token resolution are unit-tested in tests/test_roc.py.
    def _generate_with_scores(self, messages: list[dict], max_new_tokens: int = 4):
        """Like _generate but also returns the first generated step's logits (1-D)."""
        proc = self._processor
        images = []
        for m in messages:
            for blk in m.get("content", []):
                if isinstance(blk, dict) and blk.get("type") == "image":
                    images.append(blk["image"])
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt_text], images=images if images else None,
                      return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        import torch
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        text = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        step0_logits = out.scores[0][0].float().cpu().numpy()   # (vocab,) for batch item 0
        return text, step0_logits

    def two_turn_yes_prob(
        self,
        sample_image: Path,
        sample_prompt: str,
        test_image: Path,
        test_prompt: str,
        system: str,
        max_new_tokens: int = 4,
    ) -> tuple[str, float]:
        """Two-turn change detection returning (decoded_text, P(YES)) for the ROC sweep."""
        from ..analysis.roc import resolve_yes_no_token_ids, yes_prob_from_scores
        self.ensure_loaded()
        sample = Image.open(sample_image).convert("RGB")
        test = Image.open(test_image).convert("RGB")
        m1 = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample},
                {"type": "text", "text": sample_prompt},
            ]},
        ]
        ack = self._generate(m1, max_new_tokens=16) or "READY"
        m2 = m1 + [
            {"role": "assistant", "content": [{"type": "text", "text": ack}]},
            {"role": "user", "content": [
                {"type": "image", "image": test},
                {"type": "text", "text": test_prompt},
            ]},
        ]
        text, step0_logits = self._generate_with_scores(m2, max_new_tokens=max_new_tokens)
        tok = getattr(self._processor, "tokenizer", self._processor)
        yes_ids, no_ids = resolve_yes_no_token_ids(tok)
        p_yes = yes_prob_from_scores(step0_logits, yes_ids, no_ids)
        return text, p_yes
