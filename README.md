# MLX-vs-Pytorch
Benchmarks comparing the two frameworks on Apple Silicon GPUs

# Use Black for formatting

# TODO:

------ M3 MAX ---------
For training:
1. Benchmark a simple transformers model from MLX example
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
Switch test: ran 10 times
Framework: pytorch
	Average: 146.35703275203704s - Median: 146.41792500019073s

Framework: mlx
	Average: 140.5102721452713s - Median: 140.51127195358276s

------ M1 PRO ---------
For training:
1. Benchmark a simple transformers model from MLX example
LLM train test: ran 3 times
Framework: pytorch
	Average: 1806.639595190684s - Median: 1818.6110489368439s

Framework: mlx
	Average: 1157.0066788196564s - Median: 1154.6633532047272s
2. Benchmark training MINI BERT (from cohere)
Framework: pytorch
	Average: 780.7687385876974s - Median: 765.1663811206818s

Framework: mlx
	Average: 717.1385540962219s - Median: 718.111958026886s
For inference:
1. Benchmark whisper (native)
Framework: pytorch
	Average: 31.998124384880064s - Median: 31.96485936641693s

Framework: mlx
	Average: 8.509361457824706s - Median: 8.509169936180115s
2. Benchmark TinyLLama
Framework: pytorch
	Average: 59.274635887146s - Median: 55.8025221824646s

Framework: mlx
	Average: 33.38054447174072s - Median: 33.322925329208374s
Extra:
1. Switch between CPU and GPU
Switch test: ran 10 times
Framework: pytorch
	Average: 349.7299320459366s - Median: 349.9100536108017s

Switch test: ran 10 times
Framework: mlx
	Average: 270.1572776556015s - Median: 271.8326184749603s

Benchmark on:
All connected to power and with performance mode enabled
1. M1 Pro - 16 GPU cores
2. M3 Max - 40 GPU cores

Housekeeping:
1. Ideally, I would have a script to download and set up the models :)
2. Add requirements.txt

# Can test with the latest main from both MLX and pytorch. This will give more
# trustworthy result for future usage.
