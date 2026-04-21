"""
Run one experiment: build model, train, evaluate on validation set, log result.

Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard

The first positional argument is the experiment description (a short human-
readable label). Flags are order-independent.
"""
import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result


def get_git_hash():
    """Short git SHA, or 'no-git' if we're not inside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load data (FROZEN — train + val only, never the test set)
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, "
          f"{len(feature_names)} features")

    # 2. Build model (EDITABLE — this is the only piece the agent changes)
    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate (FROZEN metric — macro F1 on validation)
    val_f1, val_acc, val_recall = evaluate(model, X_val, y_val)
    print(f"val_f1_macro: {val_f1:.6f}")
    print(f"val_accuracy: {val_acc:.6f}")
    print(f"val_recall:   {val_recall:.6f}")

    # 5. Log
    experiment_id = get_git_hash()
    log_result(experiment_id, val_f1, val_acc, val_recall,
               train_time, status, description)
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()
