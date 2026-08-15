import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.yaml"
MODEL = ROOT / "ultralytics/cfg/models/11/yolo11-AIFI-C3k2_DEConv-ContextGuidedBlock_Down-AT.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCM-YOLO on the approved clinical tongue dataset.")
    parser.add_argument("--data", default=str(DATA), help="Dataset YAML path.")
    parser.add_argument("--model", default=str(MODEL), help="Model YAML or checkpoint path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name", default="tcm_yolo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        close_mosaic=10,
        resume=False,
        project=args.project,
        name=args.name,
        single_cls=False,
        cache=False,
    )
