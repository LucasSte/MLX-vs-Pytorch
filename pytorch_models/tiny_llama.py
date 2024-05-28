from transformers import AutoTokenizer
import transformers
import torch
import pathlib
import os


class TorchLLama:
    def __init__(
        self,
        max_tokens: int,
        filename: str,
    ):
        self.max_tokens = max_tokens
        self.filename = filename

        model = "TinyLlama-1.1B-Chat-v1.0"
        current_dir = pathlib.Path(__file__).parent.resolve()
        save_dir = os.path.join(current_dir, model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=save_dir,
            torch_dtype=torch.float32,
            device_map=torch.device("mps"),
        )
        self.file = open(filename, "w")

    def generate_and_save(self, prompt: str):
        formatted_prompt = (
            f"<|system|>\nYou are a friendly chatbot who always responds wisely</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            "<|assistant|>"
        )
        seqs = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )
        print(seqs[0]["generated_text"], file=self.file, flush=True)

    def finish(self):
        self.file.close()
