# MLX-vs-PyTorch

This repository contains benchmarks for comparing two popular artificial
intelligence frameworks that work on Apple Silicon devices: MLX and PyTorch.

The idea behind this simple project is to enable a wise choice when starting an
AI project on an Apple computer.

We ran five benchmarks several times to emulate a day-to-day usage. For more information about
them, please refer to section [Details about each benchmark](#details-about-each-benchmark).

1. Training a transformers language model (`lm_train.py`).
2. Training/fine-tuning BERT (`bert_fine_tune.py`).
3. Inference using OpenAI's whisper model (`whisper_inference.py`).
4. Language model inference using TinyLLama (`llm_inference.py`).
5. A synthetic benchmark that moves data between CPU and GPU for 
   matrix multiplication (`switch_test.py`).


## Results

We executed the tests for ten iterations each, except the language model training
and the BERT training ones, for which we ran only three iterations due to the
extra time they took.

The results on the tables below show the average time for the iterations we ran.
For information about the median of the execution times for each benchmark, refer
to [raw_results.txt](raw_results.txt).

<table>
<thead>
<tr>
<th colspan="4">M1 Pro (10 CPU core, 16 GPU core, 32 GB RAM) </th>
</tr>
</thead>
    <thead>
        <tr>
            <th>Benchmark</th>
            <th>PyTorch time (s)</th>
            <th>MLX time (s)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training a transformer <br> language model</td>
            <td> 1806.63 </td>
            <td> 1157.00 </td>
        </tr>
        <tr>
            <td>Training BERT</td>
            <td> 751.02 </td>
            <td> 718.35 </td>
        </tr>
        <tr>
            <td>Whisper inference</td>
            <td> 31.99 </td>
            <td> 8.50 </td>
        </tr>
        <tr>
            <td>TinyLLama inference</td>
            <td> 59.27 </td>
            <td> 33.38 </td>
        </tr>
        <tr>
            <td>CPU/GPU switch</td>
            <td> 349.72 </td>
            <td> 270.15 </td>
        </tr>
    </tbody>
</table>

<table>
<thead>
<tr>
<th colspan="4">M3 Max (16 CPU core, 40 GPU core, 48 GB RAM) </th>
</tr>
</thead>
    <thead>
        <tr>
            <th>Benchmark</th>
            <th>PyTorch time (s)</th>
            <th>MLX time (s)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training a transformer <br> language model</td>
            <td> 912.52 </td>
            <td> 426.00 </td>
        </tr>
        <tr>
            <td>Training BERT</td>
            <td> 550.29 </td>
            <td> 408.45 </td>
        </tr>
        <tr>
            <td>Whisper inference</td>
            <td> 17.90 </td>
            <td> 4.85 </td>
        </tr>
        <tr>
            <td>TinyLLama inference</td>
            <td> 36.18 </td>
            <td> 15.41 </td>
        </tr>
        <tr>
            <td>CPU/GPU switch</td>
            <td> 146.35 </td>
            <td> 140.51 </td>
        </tr>
    </tbody>
</table>


## How to run the benchmarks

First, make sure you have git LFS installed so that you can configure your repository:

```
pip3 install -r requirements.txt
cd pytorch_models
./configure.sh
cd .. 
cd mlx_models
./configure.sh
```

Every Python file in the root folder represents a different benchmark. All of them require two arguments: the number
of times to run the benchmark and the framework. If you'd like to run, for example, the TinyLLama inference benchmark
ten times using PyTorch, execute:

```
python3 llm_inference.py --framework pytorch --iter 10
```

When the command finishes, it will print on the terminal the average and median times of the ten iterations.

### Additional settings

The `lm_train.py` benchmark needs the `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable set to zero when used with
PyTorch.

The `whisper_inference` benchmark only works with the latest commit from the PyTorch repository, so build it from 
sources to run this benchmark.

##  Details about each benchmark

### Training a transformers langauge model

For this benchmark, we copied the model from MLX's [TransformerLM example](https://github.com/ml-explore/mlx-examples/blob/a7598e9456c6455a07ff4905712c2ea3cfcd52db/transformer_lm/main.py#L15).
For the PyTorch version, we utilized the closest functions available to properly replicate the model in another framework.
The dataset utilized is the [PTB corpus](https://paperswithcode.com/dataset/penn-treebank). For more information about
the model size, epochs and other hyperparameters, refer to [lm_train.py](lm_train.py).

### Training/fine-tuning BERT

We utilized the model presented in [Conneau et al](https://arxiv.org/pdf/1705.02364), using the 
[BERT-tiny model](https://huggingface.co/prajjwal1/bert-tiny) for the respective BERT blocks. It classifies pairs of
sentences as  having a contradiction, entailment or neutral relation. It was implemented in pure PyTorch and pure 
MLX respectively. We do not initialize it with any pre-trained weights, so the benchmark can be seen as pure training.
The dataset for training was the [NLI dataset](https://sbert.net/datasets/AllNLI.tsv.gz).

The only adaptation in this case was that we used PyTorch dataloader for the MLX model too, as it was compatible with 
the tokenizer library. Even though the data loader creates a PyTorch tensor for each input, we can transform it to a 
numpy array without extra copies, so this setting did not harm the MLX results.

### Whisper inference

For the PyTorch setting, we used HuggingFace transformers library to download and execute the tiny whisper model. For 
the MLX benchmark, we used the [MLX examples tools](https://github.com/ml-explore/mlx-examples/tree/main/whisper) to
download tiny whisper and convert it to the MLX format, using `float32` as the inner data type to match that of PyTorch
(see [mlx_models/configure.sh](mlx_models/configure.sh)). The inference code for MLX leverages the `mlx_whisper` 
library.

### TinyLLama inference

For PyTorch, we downloaded the `TinyLlama-1.1B-Chat-v1.0` model from the HuggingFace repository
(see [pytorch_models/configure.sh](pytorch_models/configure.sh)), and use the transformers library to load and execute the model.

For MLX, we convert the model to the MLX format using the [MLX examples tools](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama),
and use `float32` as the data type to match that of PyTorch. We utilize the execution script from the 
[MLX examples repository](https://github.com/ml-explore/mlx-examples/blob/main/llms/llama/llama.py) with several adaptations
to account for the proper prompt formatting and execution constraints.


### CPU/GPU switch

In this benchmark, we perform matrix multiplications in a loop. First, we multiply matrices in the CPU, then we
multiply the resulting matrices in the GPU. Lastly, we reuse the results from the latter as the input
for the next iteration's CPU multiplication.

The idea behind this benchmark is to assess how effective each framework's mechanisms are to move data between
execution units. 



