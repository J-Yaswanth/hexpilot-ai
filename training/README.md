# Scene-classifier training workflow

This pipeline is independent of the existing application code. It proposes
candidate cuts using sampled HSV-histogram differences; these are suggestions,
not ground truth. Scene labels are entered by a person in the labeling UI.

## 1. Inspect videos and propose scene segments

From the project root:

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset
```

The default source is `C:\Users\USER\Downloads\viiii`. Override it with
`--video-dir`. The command writes:

- `data\labels\video_metadata.csv` with filename, duration, FPS, resolution
  (plus width and height), and frame count
- `data\labels\detected_scenes.csv` with candidate video/start/end segments
- `data\labels\annotations.csv` with one initially blank label per segment

Scene cuts are estimated from low-resolution HSV histograms sampled once per
second. Tune `--sample-interval`, `--cut-threshold`, and
`--min-scene-seconds` if cuts are too frequent or too sparse. Review the cuts
and labels before training.

## 2. Confirm labels

```powershell
.\.venv\Scripts\python.exe -m streamlit run training\label_scenes.py
```

Select each scene, inspect its three preview frames, choose a confirmed label,
and save. The tool never fills a label automatically. The allowed labels are
`conversation`, `romance`, `action`, `dance`, `fight`, `outdoor`, `indoor`,
`travel`, `crowd`, `celebration`, `dramatic`, and `other`.

## 3. Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m training.train_classifier
```

Training requires at least six labeled segments per used class, and each
class must occur in at least three distinct source videos. Every class must
also be present in train, validation, and test after grouping/splitting by
source video. If the small dataset cannot satisfy this, the script refuses to
train and explains what additional annotations are needed.

For a clearly marked, non-submission-grade prototype only, the explicit
`--experimental` option permits classes with fewer than six scenes and allows
incomplete class coverage in validation/test. Videos are still split as whole
groups, and every class must appear in training. The generated evaluation
records this mode and per-class test support; it should not be presented as a
reliable estimate of generalization.

The classifier uses pretrained ResNet18 visual features with a trained linear
classification head. The backbone is frozen, and the best head is selected by
validation macro F1. The ResNet18 ImageNet weights are downloaded by
`torchvision` on first training unless already cached. CPU training is
supported; a CUDA device is used when available.

Artifacts are saved to `models\scene_classifier\`: the model, video split,
held-out test metrics, confusion matrix, and per-scene test predictions.

## 4. Classify scenes in a new video

```powershell
.\.venv\Scripts\python.exe -m training.infer_video C:\path\to\new_video.mp4
```

Output is JSON with scene start, scene end, predicted label, and confidence.
It uses the same cut-detection defaults as dataset preparation.
