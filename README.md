<h2 align="center">MF-DETR: Multi-Frequency RT-DETR for Object Detection</h2>

---

This repository contains an MF-DETR implementation based on the Ultralytics RT-DETR codebase. The current model configuration is focused on `mf-detr-r18.yaml`.

## Updates

- Initial public release of the MF-DETR-R18 training and validation code.
- Includes custom modules used by the MF-DETR configuration:
  - `SISE`
  - `MFCF`
  - `FFAE`

## Implementations

- PyTorch code: [`ultralytics`](./ultralytics)
- Model configuration: [`ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml`](./ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml)
- Dataset YAML files: [`dataset`](./dataset)

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
