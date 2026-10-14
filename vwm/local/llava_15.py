"""LLaVA-1.5-7B 4-bit pipeline.

HF ID: "llava-hf/llava-1.5-7b-hf"
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import LocalVLM, make_4bit_config


class LLaVA15Client(LocalVLM):
    provider = "local"
    model_id = "llava-hf/llava-1.5-7b-hf"

    def _load(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=make_4bit_config(),
            device_map="auto",
        )
        self._model.train(False)

    def _generate(self, prompt_text: str, images: list[Image.Image], max_new_tokens: int) -> str:
        proc = self._processor
        inputs = proc(text=prompt_text, images=images, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        import torch
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _single_turn(self, images: Sequence[Path], prompt: str, system: str, max_new_tokens: int = 8) -> str:
        pil_images = [Image.open(p).convert("RGB") for p in images]
        # LLaVA-1.5 supports a single image per turn; we use the first.
        img = pil_images[0]
        prompt_text = (
            f"{system}\n"
            f"USER: <image>\n{prompt}\nASSISTANT:"
        )
        return self._generate(prompt_text, [img], max_new_tokens)

    def _two_turn(self, sample_image, sample_prompt, test_image, test_prompt, system, max_new_tokens=8):
        # LLaVA-1.5 is single-image-per-turn; we synthesize a multi-turn conversation by
        # passing both images and a turn-style prompt.
        sample = Image.open(sample_image).convert("RGB")
        test = Image.open(test_image).convert("RGB")
        # First, produce the ack on the sample alone.
        ack_prompt = (
            f"{system}\n"
            f"USER: <image>\n{sample_prompt}\nASSISTANT:"
        )
        ack = self._generate(ack_prompt, [sample], max_new_tokens=16) or "READY"
        # Now the full conversation with both images interleaved.
        prompt_text = (
            f"{system}\n"
            f"USER: <image>\n{sample_prompt}\nASSISTANT: {ack}\n"
            f"USER: <image>\n{test_prompt}\nASSISTANT:"
        )
        return self._generate(prompt_text, [sample, test], max_new_tokens)
