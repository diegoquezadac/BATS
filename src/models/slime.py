import numpy as np
import whisper
import nltk
import sounddevice as sd
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class SLIME:
    """
    Speech LIME.
    XAI Method for explaining speech recognition models based on LIME.
    """

    def __init__(
        self, f: Any, g: Any, sample_rate: int = 16_000, segment_length: int = 500
    ) -> None:
        
        self.f = f
        self.g = g
        # Sample rate unit: Hz
        self.sample_rate = sample_rate
        # Segment length unit: ms
        self.segment_length = segment_length

    def _transcribe(self, x: np.ndarray) -> str:
        x = whisper.pad_or_trim(x)
        result = self.f.transcribe(x)
        return result["text"]

    def _segment_audio(self, x: np.ndarray) -> np.ndarray:
        segment_length = int((self.segment_length / 1000) * self.sample_rate)
        return np.array_split(x, np.arange(segment_length, len(x), segment_length))

    def _apply_mask(self, segments, mask) -> np.ndarray:
        masked_audio = np.concatenate(
            [seg if m else np.zeros_like(seg) for seg, m in zip(segments, mask)]
        )
        return masked_audio

    def _compute_importance(self):
        assert self.coef is not None
        segment_importance = [
            coef - 2 * coef if coef < 0 else coef for coef in self.coef
        ]
        segment_importance = (segment_importance - np.min(segment_importance)) / (
            np.max(segment_importance) - np.min(segment_importance)
        )
        self.segment_importance = segment_importance

    def fit(self, x: np.ndarray, n_perturbations: int = 100):
        segments = self._segment_audio(x)
        transcription = self._transcribe(x)
        n_segments = len(segments)
        perturbation_matrix = np.random.binomial(
            1, 0.5, size=(n_perturbations, n_segments)
        )
        levenshtein_distances = np.zeros(n_perturbations)

        for i, mask in enumerate(perturbation_matrix):
            x_perturbed = self._apply_mask(segments, mask)
            y_perturbed = self._transcribe(x_perturbed)
            levenshtein_distances[i] = nltk.edit_distance(transcription, y_perturbed)

        self.X = perturbation_matrix
        self.y = levenshtein_distances
        self.n_segments = n_segments
        
        self.g.fit(self.X, self.y)

        if isinstance(self.g, LinearRegression):
            self.coef = self.g.coef_
        elif isinstance(self.g, DecisionTreeRegressor):
            self.coef = self.g.feature_importances_
        else:
            raise NotImplementedError
        
        self._compute_importance()

    def explain(self, x: np.ndarray):
        total_duration = len(x) / self.sample_rate
        segment_duration = total_duration / len(self.segment_importance)
        samples_per_segment = int(self.sample_rate * segment_duration)

        adjusted_audio = np.zeros_like(x)
        for i, importance in enumerate(self.segment_importance):
            start_idx = i * samples_per_segment
            end_idx = start_idx + samples_per_segment
            end_idx = min(end_idx, len(x))
            adjusted_audio[start_idx:end_idx] = x[start_idx:end_idx] * importance * 10

        sd.play(adjusted_audio, self.sample_rate)
