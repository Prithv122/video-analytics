"""ONNX export of RF-DETR and int8 dynamic quantisation (CPU inference)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from vidan.detect import DEFAULT_RESOLUTION, RESOLUTION_STEP

# Op types worth quantising in a ViT: the weight matmuls dominate. Conv is left in fp32
# (nine small convs in the projector; dynamic int8 Conv is poorly supported on CPU).
QUANT_OPS = ("MatMul", "Gemm")


def export_onnx(size: str, resolution: int | None, out_dir: str | Path) -> Path:
    """Export a pretrained RF-DETR (nano / small / medium) to fp32 ONNX with `RFDETR.export`.

    The graph has a static `1x3xRxR` float input and two outputs, `dets` (1, 300, 4) normalised
    cxcywh and `labels` (1, 300, 91) class logits. `resolution=None` uses the default for the
    size. Returns the path of the `.onnx` file (named by rfdetr, e.g. `rfdetr-nano.onnx`).
    """
    import rfdetr

    if size not in DEFAULT_RESOLUTION:
        raise ValueError(f"size must be one of {sorted(DEFAULT_RESOLUTION)}, got {size!r}")
    res = DEFAULT_RESOLUTION[size] if resolution is None else resolution
    if res <= 0 or res % RESOLUTION_STEP[size]:
        raise ValueError(f"resolution must be a positive multiple of {RESOLUTION_STEP[size]}")
    variant = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}[size]
    kwargs = {"device": "cpu"}
    if res != DEFAULT_RESOLUTION[size]:
        kwargs["resolution"] = res
    model = getattr(rfdetr, variant)(**kwargs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = model.export(output_dir=str(out_dir), shape=(res, res), verbose=False)
    return Path(path)


def quantize_int8(
    onnx_path: str | Path,
    out_path: str | Path,
    op_types: tuple[str, ...] = QUANT_OPS,
) -> Path:
    """Dynamic int8 quantisation: weights stored as int8, activations quantised at run time.

    Only initialiser-weighted `MatMul` / `Gemm` nodes are rewritten by default; the
    activation-by-activation matmuls inside attention stay fp32. No calibration data needed.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(onnx_path),
        str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(op_types),
    )
    return out_path


def op_type_counts(onnx_path: str | Path) -> Counter[str]:
    """Node count per op type of an ONNX graph, for reporting what a quantiser can touch."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)
