from transformers import AutoTokenizer
import transformers
import torch
import pathlib
import os


class TorchLLama:
    def __init__(self, max_tokens: int, temp: float, filename: str):
        self.max_tokens = max_tokens
        self.temp = temp
        self.filename = filename

        model = "TinyLlama-1.1B-Chat-v0.4"
        current_dir = pathlib.Path(__file__).parent.resolve()
        save_dir = os.path.join(current_dir, model)
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=save_dir,
            torch_dtype=torch.float32,
            device_map=torch.device("mps"),
        )

        self.file = open(filename, "w")

    def generate_and_save(self, prompt: str):
        eos_token_id = 32002
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        seqs = self.pipeline(
            formatted_prompt,
            do_sample=False,
            # top_k=10,
            # top_p=0.9,
            num_return_sequences=1,
            # repetition_penalty=1.1,
            max_new_tokens=1024,
            # eos_token_id=eos_token_id,
        )
        print(seqs[0]["generated_text"], file=self.file, flush=True)

    def finish(self):
        self.file.close()
