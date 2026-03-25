"""校园自行车检测系统入口脚本

用法：
  # 实时摄像头（默认摄像头 0）
  python -m campus_bike_detection.main

  # 指定摄像头编号
  python -m campus_bike_detection.main --source 1

  # 视频文件
  python -m campus_bike_detection.main --source path/to/video.mp4

  # 指定模式
  python -m campus_bike_detection.main --mode both

  # 指定设备
  python -m campus_bike_detection.main --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from campus_bike_detection import BikeDetectionSystem, CountLine, SystemConfig

# 模型文件与本脚本同目录
_DEFAULT_MODEL = str(Path(__file__).parent / "yolov8n.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校园自行车检测系统")
    parser.add_argument(
        "--source",
        default="C:/Users/admin/Desktop/cycle/data/IMG_1260.MP4",
        help="输入源：摄像头编号（如 0）或视频文件路径（如 video.mp4）",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="模型文件路径（.pt / .onnx / .tflite）",
    )
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "dynamic", "both"],
        help="运行模式：static=停放检测, dynamic=车流量, both=两者",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "npu"],
        help="推理设备（默认 cpu，有 GPU 可改为 cuda）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=300,
        help="目标处理帧率",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="检测置信度阈值",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # source：数字字符串转 int（摄像头），否则保持字符串（文件路径）
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    config = SystemConfig(
        source=source,
        model_path=args.model,
        mode=args.mode,
        device=args.device,
        target_fps=args.fps,
        enable_visualization=True,
        count_lines=[
            # 画面水平中线作为默认计数线
            CountLine("main", (0.05, 0.5), (0.95, 0.5), "both")
        ],
    )

    print(f"启动检测系统")
    print(f"  输入源  : {source}")
    print(f"  模型    : {args.model}")
    print(f"  模式    : {args.mode}")
    print(f"  设备    : {args.device}")
    print(f"  目标FPS : {args.fps}")
    print(f"  置信度  : {args.conf}")
    print("按 'q' 退出\n")

    with BikeDetectionSystem(config) as system:
        # 将置信度阈值传给推理引擎
        system.engine.conf_thresh = args.conf
        report = system.run()

    print("\n===== 会话报告 =====")
    print(f"总帧数      : {report.total_frames}")
    print(f"峰值停放数  : {report.peak_parked_count}")
    print(f"平均停放数  : {report.avg_parked_count:.1f}")
    print(f"累计检测数  : {report.total_detected_count}  (唯一自行车，track ID 去重)")
    print(f"各线路流量  : {report.total_flow_by_line}")
    duration = report.end_time - report.start_time
    print(f"运行时长    : {duration:.1f}s")


if __name__ == "__main__":
    main()
