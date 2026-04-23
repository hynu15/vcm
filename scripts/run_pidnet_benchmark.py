import argparse

from model_benchmark_common import run_full_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline for PIDNet model")
    parser.add_argument("--model-name", type=str, default="pidnet_s")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-height", type=int, default=512)
    parser.add_argument("--train-width", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-frames", type=int, default=20)
    parser.add_argument("--crf-roi", type=int, default=25)
    parser.add_argument("--crf-non", type=int, default=32)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--latency-frames", type=int, default=10)
    parser.add_argument("--latency-height", type=int, default=384)
    parser.add_argument("--latency-width", type=int, default=768)
    parser.add_argument("--latency-preset", type=str, default="ultrafast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_full_benchmark(args)
    print("[PIDNet] Benchmark done")
    print(f"  training_csv: {result.training_csv}")
    print(f"  training_summary_json: {result.training_summary_json}")
    print(f"  metrics_per_frame_csv: {result.metrics_per_frame_csv}")
    print(f"  metrics_summary_csv: {result.metrics_summary_csv}")
    print(f"  latency_csv: {result.latency_csv}")
    print(f"  checkpoint_path: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
