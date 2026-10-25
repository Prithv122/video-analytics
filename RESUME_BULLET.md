# Resume Bullets — video-analytics

Form: **action → technical specifics → measured outcome.** Every number is in `results/`.

---

## Bullets

- Built a detect → ByteTrack → line-count pipeline (RF-DETR on ONNX Runtime, CPU) and scored every
  crossing against annotated tracks on held-out MOT20 under a pre-registered plan; identity-aware
  matching and per-error attribution showed misses, not identity switches, drive the error
  (recall ≤ 0.19 in all 32 configurations, switches ≤ 5 % of errors).
- Found that a confidence threshold tuned on MOT17 produced zero tracks on MOT20's densest
  sequence (peak person score 0.59 vs threshold 0.60). Benchmarked CPU inference behind a
  quiet-machine gate: int8 1.38× faster than fp32, ONNX Runtime 0.70–0.84× eager PyTorch.

## Roles

- **AI / CV engineer:** the pipeline, ONNX export and int8 quantisation, the measured CPU budget.
- **Data scientist / ML engineer:** pre-registration, dev/test discipline, cluster bootstrap, error attribution.

## Keywords

Multi-object tracking · ByteTrack · RF-DETR · ONNX Runtime · int8 quantisation · MOT17/MOT20 ·
MOTA/IDF1 · event matching · cluster bootstrap · pre-registration · domain shift
