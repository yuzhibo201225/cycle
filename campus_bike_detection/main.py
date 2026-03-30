from __future__ import annotations

import argparse
from pathlib import Path

from campus_bike_detection.models import CountLine, SystemConfig
from campus_bike_detection.system import BikeDetectionSystem

DEFAULT_MODEL = str(Path(__file__).parent / "yolov8n.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Campus Bike Real-time Detection")
    parser.add_argument("--source", default="0", help="camera id or video file path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto", choices=["auto", "pt", "onnx", "trt"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    cfg = SystemConfig(
        source=source,
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=not args.no_show,
        line=CountLine("main", (0.05, 0.5), (0.95, 0.5)),
    )

    with BikeDetectionSystem(cfg) as system:
        report = system.run()

    print("\n=== Session Report ===")
    print(f"Frames      : {report.total_frames}")
    print(f"Avg FPS     : {report.avg_fps:.2f}")
    print(f"Peak Count  : {report.peak_count}")
    print(f"Total Bikes : {report.total_count}")
    print(f"Line Counts : {report.line_counts}")


if __name__ == "__main__":
    main()
