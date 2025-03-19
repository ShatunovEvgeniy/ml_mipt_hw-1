import os
import json


from src.compute_metrics import compute_metrics


def test_compute_metrics():
    compute_metrics()
    path = "final_metrics.json"
    assert os.path.exists(path)

    with open(path, "r") as f:
        metrics = json.load(f)
    assert metrics["accuracy"] is float
