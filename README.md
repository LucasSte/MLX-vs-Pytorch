# MLX-vs-Pytorch
Benchmarks comparing the two frameworks on Apple Silicon GPUs

# Use Black for formatting

# TODO:
For training:
1. Benchmark a simple transformers model from MLX example
2. Benchmark training MINI BERT (from cohere)

For inference:
1. Benchmark whisper (native)
2. Benchmark LLama (or other LLM) through huggingface.

Extra:
1. Switch between CPU and GPU

Benchmark on:
All connected to power and with performance mode enabled
1. M1 Pro - 16 GPU cores
2. M3 Max - 40 GPU cores


# Can test with the latest main from both MLX and pytorch. This will give more
# trustworthy result for future usage.