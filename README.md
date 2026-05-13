# DA360 ONNX

Export [Insta360 DA360](https://github.com/Insta360-Research-Team/DA360) (“Depth Anything in 360°”) checkpoints to **ONNX**, and optionally run inference from **C++** with ONNX Runtime.

## Contents

| Path                           | Purpose                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/convert_da360_to_onnx.py` | PyTorch → ONNX export (needs a local clone of the official DA360 repo).                                                                                         |
| `cpp/`                         | **OpenCV + ONNX Runtime** inference: `normalized_rgb` → `pred_disp`, disparity preview PNG, optional **ASCII PLY** point cloud (ERP unprojection like DA360 `saver.py`). |
| `web/pointcloud_viewer.html`   | Minimal **Three.js** viewer: open a `.ply` file in the browser (no server required).                                                                              |

## ONNX contract

The exporter traces the same tensors as DA360’s `test.py`:

- **Input** `normalized_rgb`: float32, shape `[N, 3, H, W]`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`; see DA360 `datasets/real.py`).
- **Output** `pred_disp`: disparity; many pipelines use `1 / pred_disp` for relative depth.

`H` and `W` come from the checkpoint by default (typically **518×1036**) and must be divisible by **14**. During export, the xFormers attention path is disabled so tracing uses standard attention (better ONNX compatibility on macOS where xFormers may not install).

### Point clouds: ONNX vs post-processing

Upstream DA360 builds colored point clouds in **`saver.py`** (`save_as_point_cloud`): disparity → depth (`1 / pred_disp`, then divide by min depth like **`test.py`**), then **equirectangular-to-Euclidean** mapping for each pixel with optional masking (`pred_depth < 200`). That is **deterministic geometry**, not part of the neural network.

This repo keeps **only `pred_disp` in ONNX** (same as PyTorch export). **Depth normalization + ERP unprojection + PLY writing** run in C++ after inference when you pass **`--ply`**, matching the same formulas as `saver.py` so you do not need Open3D or Python at runtime for point clouds.

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

Run from **`cpp/`** after building (adjust paths as needed):

```bash
cd cpp
mkdir -p ../output

./build/da360_onnx_test \
  ../onnx_models/DA360_small.onnx \
  ../input/panorama.jpg \
  --preview ../output/preview.png \
  --ply ../output/cloud.ply
```

- **`--preview`** — false-color disparity heatmap (optional; default file name `pred_disp_preview.png` if you omit flags).
- **`--ply`** — optional **ASCII PLY** point cloud with RGB from the resized panorama (same ERP math as DA360 `saver.save_as_point_cloud`).
- **`--ply-flip-y`** — negate **Y** in the PLY file (see orientation note below).
- **`--ply-flip-x`** — negate **X** (try if left/right looks mirrored in 3D).

The program reads input spatial size from the ONNX model and resizes the image with cubic interpolation to match DA360’s preprocessing style.

### Point cloud orientation (why it can look “flipped”)

The C++ math matches **`saver.py`**: `Y = depth * cos(Theta)` with `Theta` decreasing from the **top** image row toward the bottom. That puts the **top of the equirectangle** (often sky / north pole of the sphere) at **negative** world **Y**. Many 3D viewers (including **Three.js** with +Y as “up”) then show the scene **vertically inverted** compared to what feels natural.

- **`web/pointcloud_viewer.html`** enables **“Flip Y for Three.js”** by default (applies `scale.y = -1` on the loaded points). Uncheck it if you already flipped at export or use a tool that matches DA360’s axes.
- For **CloudCompare, MeshLab, etc.**, either rotate the cloud in-app or re-export with **`--ply-flip-y`** so the file itself has sky at +Y.

If the texture looks **mirrored left–right** on the sphere, try **`--ply-flip-x`** or toggle **Flip X** in the web viewer.

### Viewing `.ply` in the browser (Three.js)

Open **`web/pointcloud_viewer.html`** locally (double-click or “Open with” a browser). Click **Open ASCII PLY** and choose the file produced by **`--ply`**. Uses **Three.js** + **`PLYLoader`** + **`OrbitControls`** from `cdn.jsdelivr.net` (loads online — needs internet once).

**Flip Y for Three.js** is checked by default (see [Point cloud orientation](#point-cloud-orientation-why-it-can-look-flipped)). **Flip X** is optional for left–right mirroring.

No local server is required because the page uses a **file input** (not `fetch`). Large clouds (~400k points) may take a few seconds to parse.

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
