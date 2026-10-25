# Interview Prep — video-analytics

Five questions, five answers. Every number here is in `results/` and README §5.

---

### Q1. Walk me through the architecture in 90 seconds.

**A:** Two stages. The detect stage runs RF-DETR (COCO weights, person class only) once per
sequence on ONNX Runtime and caches every box scoring at least 0.10. Everything else replays from
that cache on CPU: ByteTrack from the `trackers` package, then a line counter written as one state
machine (the naive rule is the same machine with a zero-width band), then evaluation. Ground truth
goes through the same counter, so the counting rule can't masquerade as accuracy. Each predicted
crossing is matched to a true one only if the direction agrees, it is within one second, **and**
the predicted track is on the person who actually crossed. Every unmatched crossing gets a cause:
duplicate, spurious or unmatched for the extras; tracked-not-counted, fragmented or missed for the
misses. Settings were tuned on three MOT17 sequences, the selection was committed to git, and the
MOT20 test set was scored once behind a guard that refuses a second run.

### Q2. Why identity-aware matching instead of comparing counts?

**A:** Because a count can be right for the wrong reasons. If the tracker misses one person and
double-counts another a second later, the hourly total is exact and both errors are invisible.
Time-only event matching still hides it in a crowd, because the miss and the duplicate fall
within each other's tolerance window. So a predicted crossing only matches if the predicted
track's majority ground-truth identity around that moment is the person who crossed. I registered
H1 to measure how much a net count hides. It failed on the test set, and the reason is
instructive: the count wasn't exact-but-wrong, it was simply far too low (WAPE 0.94). Cancellation
needs both kinds of error, and on MOT20 almost all errors were misses.

### Q3. What's the weakest part, and what breaks first?

**A:** The detector's confidence under domain shift. Dev tuning picked τ = 0.6 for the reference
model. On MOT20-05, with 195 annotated people per frame, that model's highest person score in
3,315 frames was 0.59, so the tracker never started a track and both lines counted zero, with no
error raised. Across all 32 configurations recall never exceeded 0.19, and f1 tracked the selected
threshold (an observation: the threshold was chosen per configuration, so it's confounded with
the rest). Second weakness: the anchor. On dev, the
bottom of a visible-part box isn't a foot. When legs are occluded the box shrinks and its bottom
jumps across the line and back. With the band on, that pattern accounted for 34 of the 45 extra
dev crossings that were left. At scale I'd count on
keypoints or a ground-plane footprint, and I'd alarm on "zero tracks for N minutes".

### Q4. How do you know it works? What did you measure, against what?

**A:** Against annotated MOT20 tracks the method never saw, with a frozen plan and four
pre-registered claims, reported as they came out. H1 (net count hides error): not supported. H2
(hysteresis beats naive): not supported, reversed, −0.032 [−0.052, −0.015]; the band also needs
a track to stay on each side, which my guess is that short, fragmented tracks rarely do; I didn't
test that mechanism. H3 (int8 is a free speed-up): supported, 1.38× FPS for Small, but the
accuracy half most likely passed because dev picked a lower threshold for int8, not because int8
detects better; their score percentiles differ by only 0.009–0.016. H4
(ONNX Runtime ≥ 1.5× PyTorch): not supported; on this CPU ONNX was 0.84× (Nano) and 0.70× (Small)
as fast. Timings ran behind a gate that waited for system CPU under 15 % before every repeat.
Before the test run an independent review found three bugs that would have changed numbers
(unclipped ground truth, a matching order bug, missing FPS at frame skip). They were fixed under a
dated amendment before any tuning.

### Q5. Your best configuration scores 0.27 F1. Isn't that just a bad model?

**A:** It's an off-the-shelf COCO detector on a crowd benchmark it was never trained for: the
best cell is Nano at 384 px input on 1,173–1,920 px frames. The point was to measure that honestly
rather than tune until it looked good. The result is useful because it says where the error lives.
Identity switches are under 5 % of errors everywhere, so a better tracker is unlikely to help much.
The candidates are a lower threshold, tiled or higher-resolution inference, or a crowd-trained
detector, none of which I've tested. Each of those is a separate
pre-registered experiment. I didn't run any of them on this test set, because then the 0.27 would
be a tuned number.

---

## 30-second pitch

People counters report one number per hour, and nobody checks why it's wrong. I built a
detect-track-count pipeline on CPU and scored every predicted crossing against annotated tracks
on held-out crowded video, giving each error a cause. Tuned on MOT17 and scored once on MOT20, the
answer was clear: the detector misses people (recall ≤ 0.19 in every configuration), the tracker
almost never swaps them (≤ 5 % of errors), and a threshold tuned on sparser scenes silently
produced zero tracks on the densest one. On the speed side, int8 gave 1.38× on Small (1.27× on
Nano), and ONNX Runtime was slower than PyTorch on this CPU.
