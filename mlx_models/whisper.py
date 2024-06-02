import mlx_whisper
from datasets import load_dataset
import mlx.core as mx
import pathlib
import os


class MLXWhisper:
    def __init__(self, num_examples: int, filename: str):
        self.dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        self.file = open(filename, "w")
        model = "whisper_tiny_fp32"
        current_dir = pathlib.Path(__file__).parent.resolve()
        self.model_dir = os.path.join(current_dir, model)

        self.num_examples = 1
        # This serves to load the model
        self.generate()
        self.file.truncate(0)

        self.num_examples = num_examples

    def generate(self):
        mx.set_default_device(mx.gpu)
        for i in range(0, self.num_examples):
            audio_sample = self.dataset[i]["audio"]
            waveform = audio_sample["array"]
            text = mlx_whisper.transcribe(
                waveform, path_or_hf_repo=self.model_dir, fp16=False
            )["text"]
            print(text, file=self.file, flush=True)

    def finish(self):
        self.file.close()
