# MLX-vs-Pytorch
Benchmarks comparing the two frameworks on Apple Silicon GPUs

# Use Black for formatting

# TODO:
For training:
1. Benchmark a simple transformers model from MLX example
M3 MAX:
5 epochs, 3 iter
Framework: pytorch
	Average: 912.5205717086792s - Median: 924.3736660480499s

Framework: mlx
	Average: 426.00355768203735s - Median: 426.1944200992584s

2. Benchmark training MINI BERT (from cohere)
Ran 3 times
Framework: pytorch
	Average: 536.9084160327911s - Median: 537.460841178894s


Framework: mlx
	Average: 409.2550208568573s - Median: 409.42770981788635s

For inference:
1. Benchmark whisper (native)
Whisper inference test: ran 10 times
Framework: pytorch
	Average: 17.909158730506896s - Median: 17.877812027931213s

Framework: mlx
	Average: 4.8507798433303835s - Median: 4.839159846305847s
2. Benchmark TinyLLama
Framework: pytorch
	Average: 36.182030129432675s - Median: 34.26037609577179s

LLM inference test: ran 10 times
Framework: mlx
	Average: 15.41469841003418s - Median: 15.389396786689758s

Extra:
1. Switch between CPU and GPU

Benchmark on:
All connected to power and with performance mode enabled
1. M1 Pro - 16 GPU cores
2. M3 Max - 40 GPU cores

Housekeeping:
1. Ideally, I would have a script to download and set up the models :)
2. Add requirements.txt

# Can test with the latest main from both MLX and pytorch. This will give more
# trustworthy result for future usage.
