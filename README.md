# MLX-vs-PyTorch

This repository contains benchmarks for comparing two popular artificial
intelligence frameworks that work on Apple Silicon devices: MLX and PyTorch.

The idea behind this simple project is to enable a wise choice when starting an
AI project in an Apple computer.

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
            <td> 780.76 </td>
            <td> 717.13 </td>
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
            <td> 536.90 </td>
            <td> 409.20 </td>
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
ten times using pytorch, run:

```
python3 llm_inference.py --framework pytorch --iter 10
```

When the command finishes, it will print on the terminal the average and median times of the ten iterations.

### Additional settings

The `lm_train.py` benchmark needs the `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable set to zero when used with
pytorch.

The `whisper_inference` benchmark only works with the latest commit from the Pytorch repository, so build it from 
sources to run this benchmark.

##  Details about each benchmark