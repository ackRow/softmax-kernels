# Compare Softmax CUDA/Triton implementations

> The following code was tested using the docker image: `nvidia/cuda:12.4.0-devel-ubuntu22.04` on a Geforce RTX 2070

## Usage

* Build Python library with CUDA bindings

```bash
cd cuda
pip install .
```

* Test both implementations against Pytorch baseline

```bash
python3 assertions.py
```

* Run a benchmark

```bash
python3 benchmark.py
```

* Profile both implementations

```bash
ncu --set full [-o output_path] python3 -O assertions.py
```