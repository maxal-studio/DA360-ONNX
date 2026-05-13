# DA360 ONNX

Export [Insta360 DA360](https://github.com/Insta360-Research-Team/DA360) (“Depth Anything in 360°”) checkpoints to **ONNX**, and optionally run inference from **C++** with ONNX Runtime.

## Contents

| Path                           | Purpose                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/convert_da360_to_onnx.py` | PyTorch → ONNX export (needs a local clone of the official DA360 repo).                                                                                         |
| `cpp/`                         | Sample **OpenCV + ONNX Runtime** tester: load an image, preprocess like DA360’s `Real` dataset, run `normalized_rgb` → `pred_disp`, save a false-color preview. |

## ONNX contract

The exporter traces the same tensors as DA360’s `test.py`:

- **Input** `normalized_rgb`: float32, shape `[N, 3, H, W]`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`; see DA360 `datasets/real.py`).
- **Output** `pred_disp`: disparity; many pipelines use `1 / pred_disp` for relative depth.

`H` and `W` come from the checkpoint by default (typically **518×1036**) and must be divisible by **14**. During export, the xFormers attention path is disabled so tracing uses standard attention (better ONNX compatibility on macOS where xFormers may not install).

## Python setup

1. Python **3.12+** recommended.
2. Virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

3. Clone the upstream DA360 repository next to this project (or anywhere). The exporter needs **`--da360_root`** pointing at that folder so Python can import `networks` and `depth_anything_v2`. That clone is **large** once you add checkpoints; it is **gitignored** here (`/DA360/`).

```bash
git clone https://github.com/Insta360-Research-Team/DA360.git
```

Optionally install DA360’s own `requirements.txt` if you hit missing modules; on **Apple Silicon**, prefer PyTorch’s **macOS** wheels and skip CUDA-only index URLs from their file.

4. Download checkpoints (see DA360 README): e.g. `bash DA360/scripts/download_models.sh` inside `DA360/`.

## Export to ONNX

The exporter lives at **`src/convert_da360_to_onnx.py`** (next to the **`cpp/`** folder). If your terminal is inside **`cpp/`**, running `python src/...` looks for **`cpp/src/`**, which does not exist — you get `can't open file '.../cpp/src/convert_da360_to_onnx.py'`.

**Fix:** `cd ..` to the repo root, **or** stay in `cpp/` and use **`../`** everywhere below.

### Install Python dependencies first

The script needs **`onnx`**, **`torch`**, etc. Your conda **`(base)`** env usually does **not** include them until you install.

From the repo root, either use this project’s venv (recommended in [Python setup](#python-setup)):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Or install into whatever environment you use (conda base or another env):

```bash
pip install -r requirements.txt
```

Then run **`python src/convert_da360_to_onnx.py`** with that same activated environment.

### From repository root (recommended)

```bash
cd /path/to/da360_onnx    # folder that contains both src/ and cpp/

source .venv/bin/activate   # skip if you installed deps another way

mkdir -p onnx_models      # optional; script creates parent dirs for --output

python src/convert_da360_to_onnx.py \
  --da360_root="./DA360" \
  --model_path="./DA360/checkpoints/DA360_large.pth" \
  --output="onnx_models/DA360_large.onnx" \
  --batch_size=1
```

Useful flags: `--height`, `--width`, `--net`, `--opset`, `--device` (`auto`, `cpu`, `mps`, `cuda`).

### Apple Silicon (M1 / M2 / M3)

There is no CUDA; PyTorch uses **CPU** or **MPS**. Install PyTorch for macOS from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Do not use DA360’s CUDA-only pip index on a Mac.

- Default **`--device auto`** uses MPS when available, otherwise CPU. If ONNX export fails on MPS, use **`--device cpu`**.
- **`pretrained model is not available.`** from DA360 code is a harmless message when the Depth Anything V2 backbone file is not at the authors’ hard-coded path; your `.pth` still carries the weights needed for export.

Example (small checkpoint, CPU tracing):

```bash
PROJECT=/path/to/this/repo
source "$PROJECT/.venv/bin/activate"

mkdir -p "$PROJECT/onnx_models"

python "$PROJECT/src/convert_da360_to_onnx.py" \
  --da360_root="$PROJECT/DA360" \
  --model_path="$PROJECT/DA360/checkpoints/DA360_small.pth" \
  --output="$PROJECT/onnx_models/DA360_small.onnx" \
  --device cpu
```

## C++ smoke test (`cpp/`)

Build (Homebrew on Apple Silicon shown; adjust **`ORT_ROOT`** if ONNX Runtime is elsewhere):

```bash
brew install cmake opencv onnxruntime
cd cpp
cmake -S . -B build
cmake --build build -j
```

If `libonnxruntime.dylib` is not under `/opt/homebrew/lib`:

```bash
cmake -S . -B build -DORT_ROOT="$(brew --prefix onnxruntime)"
```

Run from **`cpp/`** after building (adjust the `.onnx` name if you used a different checkpoint):

```bash
cd cpp
./build/da360_onnx_test \
  ../onnx_models/DA360_small.onnx \
  ../input/panorama.jpg \
  ../output/preview.png
```

The program reads input spatial size from the ONNX model and resizes the image with cubic interpolation to match DA360’s preprocessing style.

## Troubleshooting

| Issue                                                    | What to try                                                                                                                         |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Export fails on MPS                                      | `--device cpu`                                                                                                                      |
| OOM on large checkpoint                                  | smaller `.pth`, or more RAM                                                                                                         |
| `Missing .../networks/da360.py`                          | `--da360_root` must point at the **root** of the cloned DA360 repo (`networks/`, `depth_anything_v2/`).                             |
| `can't open file '.../cpp/src/convert_da360_to_onnx.py'` | Run `python src/convert_da360_to_onnx.py` from the **repo root**, not from `cpp/`.                                                  |
| `ModuleNotFoundError: No module named 'onnx'`            | Install deps: `pip install -r requirements.txt` in the **same** env you use to run Python (e.g. `source .venv/bin/activate` first). |

## License

See `LICENSE`. DA360 upstream has its own license in the cloned repository.
