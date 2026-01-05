import argparse
import json
import os
from collections import defaultdict


def calculate_accuracy(cache_file: str) -> None:
    if not os.path.exists(cache_file):
        raise FileNotFoundError(cache_file)

    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    with open(cache_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            model_name = data.get("model_name", "unknown")
            language = data.get("language", "unknown")
            is_solved = bool(data.get("solved", False))

            temperature = data.get("temperature", None)
            if temperature is None:
                model_display = model_name
            else:
                model_display = f"{model_name}@temp={temperature}"
            
            use_context = bool(data.get("use_context", False))
            hint_type = str(data.get("hint_type")) if data.get("hint_type") else "None"
            pass_at = bool(data.get("pass_at_enabled", False))
            num_pass = int(data.get("num_pass", 1))

            pass_str = f"True({num_pass})" if pass_at else "False"
            config_key = (
                f"{model_display:<20} | "
                f"Ctx: {str(use_context):<5} | "
                f"Hint: {hint_type:<13} | "
                f"Pass@: {pass_str}"
            )

            stats[config_key][language]["total"] += 1
            if is_solved:
                stats[config_key][language]["correct"] += 1

    header = f"{'CONFIGURATION':<75} | {'TASK':<6} | {'ACCURACY':<8} | {'COUNTS'}"
    print("\n" + "=" * 110)
    print(header)
    print("=" * 110)

    for config in sorted(stats.keys()):
        lang_data = stats[config]
        
        for lang in sorted(lang_data.keys()):
            counts = lang_data[lang]
            total = counts["total"]
            correct = counts["correct"]
            acc = (correct / total) * 100 if total else 0.0
            print(f"{config:<75} | {lang:<6} | {acc:6.2f}%  | {correct}/{total}")
        print("-" * 110)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", help="JSONL file produced by main.py")
    args = p.parse_args()
    calculate_accuracy(args.results)


if __name__ == "__main__":
    main()