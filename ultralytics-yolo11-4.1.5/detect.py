import argparse
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs/train/exp/weights/best.pt"
DEFAULT_SOURCE = ROOT / "dataset/images/test"


def parse_args():
    parser = argparse.ArgumentParser(description="Run TCM-YOLO inference on clinical tongue images.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained model checkpoint.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, directory, video, or stream source.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        save=True,
    )
