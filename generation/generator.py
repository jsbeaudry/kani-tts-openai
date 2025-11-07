"""Text-to-speech generation logic"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from threading import Thread

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, REPETITION_CONTEXT_SIZE, MAX_TOKENS
)


class TokenIDStreamer(BaseStreamer):
    """Custom streamer that yields token IDs"""
    def __init__(self, callback):
        self.callback = callback

    def put(self, value):
        """Called by model.generate() with token IDs"""
        if len(value.shape) > 1:
            token_ids = value[0].tolist()
        else:
            token_ids = value.tolist()
        for token_id in token_ids:
            self.callback(token_id)

    def end(self):
        """Called when generation is complete"""
        pass


class TTSGenerator:
    def __init__(self):
        # Choose device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None  # we'll move it manually
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def prepare_input(self, prompt, speaker_id=None):
        """Build custom input_ids with special tokens"""
        text_prompt = f"{speaker_id.lower()}: {prompt}" if speaker_id else prompt

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids.to(self.device)
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64, device=self.device)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64, device=self.device)

        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64, device=self.device)

        return modified_input_ids, attention_mask

    def generate(
        self,
        prompt,
        audio_writer,
        max_tokens=MAX_TOKENS,
        speaker_id=None,
        temperature=TEMPERATURE
    ):
        """Generate speech tokens from text prompt"""
        modified_input_ids, attention_mask = self.prepare_input(prompt, speaker_id)

        start_time = time.time()
        all_token_ids = []

        def on_token_generated(token_id):
            all_token_ids.append(token_id)
            audio_writer.add_token(token_id)

        streamer = TokenIDStreamer(callback=on_token_generated)

        generation_kwargs = dict(
            input_ids=modified_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            eos_token_id=END_OF_AI,
            streamer=streamer,
        )

        # Launch generation in a background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        thread.join()

        end_time = time.time()

        print(f"\n[MAIN] Generation complete. Total tokens: {len(all_token_ids)}")

        generated_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)

        return {
            "generated_text": generated_text,
            "all_token_ids": all_token_ids,
            "generation_time": end_time - start_time,
            "point_1": start_time,
            "point_2": end_time,
        }
