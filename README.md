# TCM-YOLO

Source code for the paper “Interpretable Deep Learning for Objective Tongue Diagnosis: The TCM-YOLO Framework”.

## Reproducibility scope

This repository contains the Ultralytics-based implementation and the model configuration used for TCM-YOLO. The active entry points are:

- Data configuration: `ultralytics-yolo11-4.1.5/data.yaml`
- Training script: `ultralytics-yolo11-4.1.5/train.py`
- Active model configuration: `ultralytics-yolo11-4.1.5/ultralytics/cfg/models/11/yolo11-AIFI-C3k2_DEConv-ContextGuidedBlock_Down-AT.yaml`

The clinical tongue-image task has two classes: `Yang Deficiency` and `Yin Deficiency`. The implementation combines AIFI, context-guided downsampling, DEConv-based blocks, CAFM, and the Wise-SIoU loss path used by the paper.

The clinical dataset is not included in this repository. It is maintained as a restricted Zenodo record because of privacy and institutional ethics requirements: https://doi.org/10.5281/zenodo.19480854. The public TonguExpert resource used for external evaluation is available at https://www.biosino.org/TonguExpert.

## Quick start

1. Install the dependencies listed in `ultralytics-yolo11-4.1.5/requirements.txt`.
2. Place an approved local copy of the clinical dataset under `ultralytics-yolo11-4.1.5/dataset`, or edit `data.yaml` to point to an approved local path. The expected layout is `images/train`, `images/val`, and optionally `images/test`, with matching YOLO label directories.
3. From `ultralytics-yolo11-4.1.5`, run:

```bash
python train.py
```

The training entry point uses repository-relative paths and does not contain author-machine paths or bundled patient data. Reported paper metrics should be reproduced only with the paper's complete experiment protocol and an authorized dataset copy.
