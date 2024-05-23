import mlx_whisper
from datasets import load_dataset
import mlx.core as mx

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)

mx.set_default_device(mx.gpu)
# TODO: Be sure to use the fp-32 model
for i in range(0, 50):
    audio_sample = ds[i]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    text = mlx_whisper.transcribe(
        waveform, path_or_hf_repo="./whisper_tiny_fp32", fp16=False
    )["text"]
    print(text)
