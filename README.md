## MF-DETR: An Efficient End-to-End Framework for Small Object Detection in UAV Imagery

---

This repository contains the implementation of **MF-DETR** for UAV small object detection.

## Updates

- Release MF-DETR-R18.

### Experimental Results on the VisDrone-2019-DET Dataset

| **Model** | **Backbone** | **Input Size** | **Params (M)** | **GFLOPs** | **AP** | **AP$_{50}$** |
|---|---|---:|---:|---:|---:|---:|
| **MF-DETR (Ours)** | ResNet18 | 640 x 640 | **21.8** | **73.5** | **30.2** | **49.5** |

---

### Experimental Results on UAVVaste Dataset

| **Model** | **Params (M)** | **GFLOPs** | **AP** | **AP$_{50}$** |
|---|---:|---:|---:|---:|
| **MF-DETR (Ours)** | **21.8** | **73.5** | **49.3** | **80.4** |

---

### Experimental Results on TinyPerson Dataset

| **Model** | **P** | **R** | **F1** | **AP** | **AP$_{50}$** | **APsmall** | **APmedium** | **APlarge** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MF-DETR (Ours)** | **53.9** | **34.1** | **41.7** | **10.2** | **30.3** | **8.7** | **18.4** | **20.4** |

---

## Ablation Study on VisDrone

| **Model Configuration** | **AP** | **AP$_{50}$** |
|---|---:|---:|
| Baseline | 26.7 | 44.6 |
| Baseline + SISE | 28.1 | 46.3 |
| Baseline + MFCF | 29.0 | 47.9 |
| Baseline + FFAE | 28.9 | 47.5 |
| Baseline + SISE + MFCF | 29.1 | 48.0 |
| Baseline + SISE + FFAE | 29.0 | 47.8 |
| Baseline + MFCF + FFAE | 30.1 | 49.1 |
| **Full Model** | **30.2** | **49.5** |
