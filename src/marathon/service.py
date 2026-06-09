from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import onnxruntime as rt
from feast import FeatureStore


@bentoml.service(name="race_predictor")
class RacePredictor:
    def __init__(self) -> None:
        # runs once at startup: build the ONNX session ONE time and reuse it
        self.session = rt.InferenceSession(
            "artifacts/riegel.onnx", providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        feature_repo_path = str(Path(__file__).parent.parent.parent / "feature_repo")
        self.store = FeatureStore(repo_path=feature_repo_path)

    def _predict(self, distance_km: float) -> dict[str, Any]:
        log_d = np.array([[np.log(distance_km)]], dtype=np.float32)
        log_t = self.session.run(None, {self.input_name: log_d})[0]
        seconds = float(np.exp(log_t.ravel()[0]))

        online_features = self.store.get_online_features(
            features=[
                "daily_fitness:ctl",
                "daily_fitness:atl",
                "daily_fitness:tsb",
                "daily_fitness:readiness_score",
            ],
            entity_rows=[{"athlete_id": 1}],
        ).to_dict()

        return {
            "distance_km": distance_km,
            "predicted_seconds": round(seconds, 1),
            "fitness_state": {
                "ctl": online_features["ctl"][0],
                "atl": online_features["atl"][0],
                "tsb": online_features["tsb"][0],
                "readiness_score": online_features["readiness_score"][0],
            },
        }

    @bentoml.api
    def predict_race(self, distance_km: float = 10.0) -> dict[str, Any]:
        return self._predict(distance_km)

    @bentoml.api
    def predict_marathon(self) -> dict[str, Any]:
        return self._predict(42.195)
