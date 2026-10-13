#!/usr/bin/env python3
"""Regenerates cuda_sgemm_colab.ipynb from the files in this repo.

The notebook embeds the sources in %%writefile cells so it runs on Colab with
nothing else uploaded. Re-run this after changing any embedded file.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED = ["src/kernels.cuh", "src/main.cu", "scripts/profile.sh", "scripts/roofline.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md("# CUDA SGEMM: staged optimization vs cuBLAS\n\n"
       "**First:** Runtime > Change runtime type > **T4 GPU** (or any GPU), then Runtime > Run all.\n\n"
       "Stages: naive, coalesced, shared-memory tiled, register-blocked, vectorized loads."),
    md("## 1. Check the GPU and toolchain"),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv\n!nvcc --version | tail -2"),
    md("## 2. Write the sources"),
    code("import os\n"
         "os.makedirs('/content/cuda-sgemm', exist_ok=True)\n"
         "%cd /content/cuda-sgemm\n"
         "!mkdir -p src scripts build results"),
]
for path in EMBED:
    with open(os.path.join(ROOT, path)) as f:
        cells.append(code(f"%%writefile {path}\n{f.read()}"))

cells += [
    md("## 3. Build"),
    code("!nvcc -O3 -std=c++17 -lineinfo src/main.cu -o build/sgemm -lcublas && echo built"),
    md("## 4. Benchmark every stage against cuBLAS\n\n"
       "Prints GFLOP/s, percent of cuBLAS and the error against cuBLAS, and measures the roofline "
       "ceilings (DRAM bandwidth, peak FMA rate). Takes a minute or two; the naive stages are slow at N=4096."),
    code("!./build/sgemm"),
    code("import pandas as pd\n"
         "bench = pd.read_csv('results/bench.csv')\n"
         "bench.pivot(index='kernel', columns='n', values='gflops').loc[\n"
         "    ['naive', 'coalesced', 'tiled', 'regblock', 'vectorized', 'cublas']].round(1)"),
    md("## 5. Profile with Nsight Compute\n\n"
       "Collects DRAM bytes moved and floating-point instruction counts per stage. "
       "If `ncu` is missing or lacks permission on this runtime, skip this cell: the roofline "
       "below still runs, using the algorithmic intensity instead of the measured one."),
    code("!which ncu && bash scripts/profile.sh 2048 || echo 'ncu unavailable or failed; see results/ncu/*.log'"),
    md("## 6. Roofline: memory-bound vs compute-bound"),
    code("!python3 scripts/roofline.py --size 2048\n"
         "from IPython.display import Image\n"
         "Image('results/roofline_2048.png', width=760)"),
    md("## 7. (Optional) Download the results"),
    code("!zip -qr results.zip results\n"
         "from google.colab import files\n"
         "files.download('results.zip')"),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
out = os.path.join(ROOT, "cuda_sgemm_colab.ipynb")
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")
print("wrote", out)
