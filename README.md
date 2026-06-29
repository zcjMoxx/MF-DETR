<h2 align="center">MF-DETR: An Efficient End-to-End Framework for Small Object Detection in UAV Imagery</h2>

---

This repository contains the implementation of **MF-DETR**, a degradation-oriented end-to-end detection framework for UAV small-object detection.

MF-DETR is built on RT-DETR and improves small-object representation through a progressive representation recovery pipeline:

- **SISE**: Statistical Intra-scale Semantic Enhancement suppresses background-dominated token responses and strengthens weak intra-scale semantics.
- **MFCF**: Multi-scale Frequency-spatial Collaboration Fusion recovers texture, boundary, and structural details during multi-scale feature aggregation.
- **FFAE**: Frequency-guided Feature Alignment and Enhancement corrects cross-level semantic and spatial misalignment before detection.

Experiments on VisDrone show that MF-DETR improves AP and AP50 by **3.5** and **4.9** percentage points over the RT-DETR-R18 baseline.

## Updates

- Initial public release of MF-DETR-R18 training and validation code.
- Release the cleaned MF-DETR configuration with `SISE`, `MFCF`, and `FFAE`.

## Implementations

- PyTorch code: [`ultralytics`](./ultralytics)
- Model configuration: [`ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml`](./ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml)
- Dataset YAML files: [`dataset`](./dataset)

## Experimental Results on VisDrone

| Model | Backbone | Input size | Params (M) | FLOPs (G) | AP | AP50 |
|---|---|---:|---:|---:|---:|---:|
| **MF-DETR (Ours)** | ResNet18 | 640 x 640 | **21.8** | **73.5** | **30.2** | **49.5** |

## Experimental Results on UAVVaste

| Model | Params (M) | FLOPs (G) | AP | AP50 |
|---|---:|---:|---:|---:|
| **MF-DETR (Ours)** | **21.8** | **73.5** | **49.3** | **80.4** |

## Experimental Results on TinyPerson

| Model | P | R | F1 | AP | AP50 | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MF-DETR (Ours)** | **53.9** | **34.1** | **41.7** | **10.2** | **30.3** | **8.7** | **18.4** | **20.4** |

## Ablation Study on VisDrone

| Model Configuration | AP | AP50 |
|---|---:|---:|
| Baseline | 26.7 | 44.6 |
| Baseline + SISE | 28.1 | 46.3 |
| Baseline + MFCF | 29.0 | 47.9 |
| Baseline + FFAE | 28.9 | 47.5 |
| Baseline + SISE + MFCF | 29.1 | 48.0 |
| Baseline + SISE + FFAE | 29.0 | 47.8 |
| Baseline + MFCF + FFAE | 30.1 | 49.1 |
| **Full Model** | **30.2** | **49.5** |

## Training

Update the dataset path in the corresponding YAML file under `dataset/`, then run:

```bash
python train.py
```

By default, `train.py` uses:

```text
dataset/data_VisDrone.yaml
ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml
```

## Validation

After training, update the checkpoint path in `val.py` if needed, then run:

```bash
python val.py
```

## COCO Metrics

To evaluate COCO-format predictions:

```bash
python get_COCO_metrice.py --anno_json path/to/annotations.json --pred_json path/to/predictions.json
```

## Notes

Training outputs, model weights, caches, and local experiment artifacts are ignored by git. Dataset YAML files are included, but raw datasets should be stored outside this repository.
