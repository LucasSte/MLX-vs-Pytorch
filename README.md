# MLX-vs-PyTorch

This repository contains benchmarks for comparing two popular artificial
intelligence frameworks that work on Apple Silicon devices: MLX and PyTorch.

The idea behind this simple project is to enable a wise choice when starting an
AI project in an Apple computer.

We've run five benchmarks several times to emulate a day-to-day usage. For more information about
them, please refer to section (REFERENCE HERE).

1. Training a transformers language model.
2. Training/fine-tuning BERT.
3. Inference using OpenAI's whisper model.
4. Language model inference using TinyLLama.
5. A synthetic benchmark that moves data between CPU and GPU for 
   matrix multiplication.

---

## Results

TODO: Explain what the numbers mean
TODO: Showing both the mean and median clutters the table.
Present only the median!
Why not the mean and stddev?

<table>
    <thead>
        <tr>
            <th>Device</th>
            <th>Benchmark</th>
            <th>PyTorch time (s)</th>
            <th>MLX time (s)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
            <td>L3 Name D</td>
        </tr>
<tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
            <td>L3 Name D</td>
        </tr>
    </tbody>
</table>