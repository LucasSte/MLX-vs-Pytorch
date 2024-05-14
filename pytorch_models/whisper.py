from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = torch.device('cpu')
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


for i in range(0, 50):
    audio_sample = ds[i]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device)

    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    predicted_ids = model.generate(input_features.to(device))

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(transcription)
