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

TODO: For information about the median, check raw.txt

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