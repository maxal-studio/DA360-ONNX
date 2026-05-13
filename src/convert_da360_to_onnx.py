# Copyright (c) 2026 MSch8791.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export Insta360 DA360 (Depth Anything in 360°) checkpoints to ONNX.

Expects a local clone of https://github.com/Insta360-Research-Team/DA360 on disk
(the script imports `networks` and `depth_anything_v2` from that tree).
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

import onnx
import torch
import torch.nn as nn


class DA360OnnxWrapper(nn.Module):
    """Single tensor in / out for ONNX (same tensors as PyTorch test.py)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        return self.model(normalized_rgb)["pred_disp"]


def _prepend_da360_root(da360_root: str) -> str:
    root = os.path.abspath(da360_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--da360_root is not a directory: {root}")
    marker = os.path.join(root, "networks", "da360.py")
    if not os.path.isfile(marker):
        raise FileNotFoundError(
            f"Missing {marker}. Clone https://github.com/Insta360-Research-Team/DA360 "
            "and pass the repository root with --da360_root."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _torch_load_checkpoint(path: str, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_da360(
    model_path: str,
    da360_root: str,
    net_override: str | None,
) -> tuple[nn.Module, int, int]:
    _prepend_da360_root(da360_root)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    # MemEffAttention uses xFormers when available; those ops are a poor fit for ONNX.
    # At runtime this flag is checked on every forward, so patching after import is enough.
    import depth_anything_v2.dinov2_layers.attention as da360_attn

    da360_attn.XFORMERS_AVAILABLE = False

    import networks

    model_dict = _torch_load_checkpoint(model_path, map_location="cpu")
    if "net" not in model_dict:
        model_dict["net"] = "DA360"
    if "dinov2_encoder" not in model_dict:
        model_dict["dinov2_encoder"] = "vits"
    if "height" not in model_dict:
        model_dict["height"] = 518
    if "width" not in model_dict:
        model_dict["width"] = 1036
    if net_override:
        model_dict["net"] = net_override

    net_cls = getattr(networks, model_dict["net"])
    model = net_cls(
        model_dict["height"],
        model_dict["width"],
        dinov2_encoder=model_dict["dinov2_encoder"],
        mixed_precision=False,
    )
    state = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in state}, strict=False)
    model.eval()
    return model, int(model_dict["height"]), int(model_dict["width"])


def get_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a DA360 .pth checkpoint to ONNX (equirectangular normalized RGB in, pred_disp out)."
    )
    p.add_argument(
        "--da360_root",
        type=str,
        required=True,
        help="Path to the cloned DA360 repository root (contains networks/ and depth_anything_v2/).",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the DA360 checkpoint (.pth), e.g. ./checkpoints/DA360_large.pth",
    )
    p.add_argument("--output", type=str, required=True, help="Output .onnx file path.")
    p.add_argument("--batch_size", type=int, default=1, help="Dummy batch size used for tracing.")
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override input height (must match training; default from checkpoint).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override input width (must match training; default from checkpoint).",
    )
    p.add_argument("--net", type=str, default=None, help="Override architecture key in checkpoint (see DA360 test.py).")
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Device used while tracing. On Apple Silicon, 'auto' tries MPS then CPU. "
        "If ONNX export fails on MPS, pass --device cpu.",
    )
    return p.parse_args()


def _resolve_export_device(requested: str) -> torch.device:
    r = requested.lower().strip()
    if r == "cpu":
        return torch.device("cpu")
    if r == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if r == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available. Use a PyTorch build with MPS "
                "(Apple Silicon) or choose --device cpu."
            )
        return torch.device("mps")
    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device choice: {requested!r}")


def _export_onnx(wrapper: nn.Module, dummy: torch.Tensor, output_path: str, opset: int) -> None:
    sig = inspect.signature(torch.onnx.export)
    kwargs: dict = dict(
        model=wrapper,
        args=dummy,
        f=output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["normalized_rgb"],
        output_names=["pred_disp"],
        dynamic_axes={
            "normalized_rgb": {0: "batch"},
            "pred_disp": {0: "batch"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(**kwargs)


def main() -> None:
    args = get_arguments()
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    device = _resolve_export_device(args.device)

    print("Loading DA360 from checkpoint...")
    model, h_ckpt, w_ckpt = load_da360(args.model_path, args.da360_root, args.net)
    h = args.height if args.height is not None else h_ckpt
    w = args.width if args.width is not None else w_ckpt
    if h % 14 != 0 or w % 14 != 0:
        raise ValueError(
            f"height and width must be divisible by 14 (DINO patch size); got {h}x{w}."
        )

    model = model.to(device)
    wrapper = DA360OnnxWrapper(model).to(device)
    print(f"Model on {device}, trace size batch={args.batch_size}, spatial {h}x{w}")

    dummy = torch.zeros(args.batch_size, 3, h, w, device=device, dtype=torch.float32)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Exporting ONNX...")
    with torch.no_grad():
        _export_onnx(wrapper, dummy, args.output, args.opset)

    print(f"Saved ONNX to {args.output}")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("onnx.checker passed.")


if __name__ == "__main__":
    main()
