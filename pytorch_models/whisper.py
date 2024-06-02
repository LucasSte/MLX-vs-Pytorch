from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = torch.device("mps")


class TorchWhisper:
    def __init__(self, num_examples: int, filename: str):
        self.num_examples = num_examples
        self.dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny.en"
        ).to(device)
        self.file = open(filename, "w")

    def generate(self):
        for i in range(0, self.num_examples):
            audio_sample = self.dataset[i]["audio"]
            waveform = audio_sample["array"]
            sampling_rate = audio_sample["sampling_rate"]
            input_features = self.processor(
                waveform, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features
            predicted_ids = self.model.generate(input_features.to(device))
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            print(transcription[0], file=self.file, flush=True)

    def finish(self):
        self.file.close()
