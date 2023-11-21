import torch
import jiwer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Any
from typing import Tuple
from datasets import Audio
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class RepresentationErasure:
    """
    Representation Erasure.
    XAI Method for explaining speech recognition models based on Representation Erasure.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_examples: int = 0,
        text_column_name: str = "sentence",
        whisper_model: str = "openai/whisper-tiny",
        sample_rate: int = 16_000,
        max_dims_to_erase: int = 80,
    ) -> None:
        self.dataset = dataset
        self.max_examples = max_examples
        self.text_column_name = text_column_name
        self.max_dims = max_dims_to_erase
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        self.sample_rate = sample_rate
        self.model.config.forced_decoder_ids = None

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset.

        Args:
            dataset: Dataset to preprocess.

        Returns:
            Preprocessed dataset.
        """
        print("Preprocessing dataset...")
        if self.max_examples > 0:
            dataset = dataset.select(range(self.max_examples))
        dataset_resampled = dataset.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate)
        )
        dataset_shrinked = dataset_resampled.select_columns(
            ["audio", self.text_column_name]
        )
        dataset_preprocessed = dataset_shrinked.map(self._apply_processor)

        return dataset_preprocessed

    def _apply_processor(self, batch: Any) -> Any:
        """
        Apply processor to batch.

        Args:
            batch: Batch to preprocess.

        Returns:
            Preprocessed batch.
        """
        batch["input_features"] = self.processor(
            batch["audio"]["array"], sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features
        return batch

    def _apply_erasure(self, batch: Any, dims: list) -> Any:
        """
        Apply erasure to batch.

        Args:
            batch: Batch to preprocess.
            dims: Dimensions to erase.

        Returns:
            batch with dims erasured.
        """
        features = torch.asarray(batch["input_features"])
        for dim in dims:
            features[:, dim] = 0
        batch["input_features"] = features
        return batch

    def _get_reference(self, batch: Any) -> Any:
        """
        Get reference for batch.

        Args:
            batch: Batch to preprocess.

        Returns:
            Batch with reference.
        """
        batch["reference"] = self.processor.tokenizer._normalize(
            batch[self.text_column_name]
        )
        return batch

    def _get_predictions(self, batch: Any) -> Any:
        """
        Get predictions for batch.

        Args:
            batch: Batch to preprocess.

        Returns:
            Batch with predictions.
        """
        input_features = torch.asarray(batch["input_features"])
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)[0]
        transcription = self.processor.decode(predicted_ids)
        batch["transcription"] = self.processor.tokenizer._normalize(transcription)
        return batch

    def _calculate_metrics(self, reference: str, hypothesis: str) -> dict:
        """
        Calculate metrics.

        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.

        Returns:
            Metrics.
        """
        metrics = {}
        output_words = jiwer.process_words(reference, hypothesis)
        outpur_chars = jiwer.process_characters(reference, hypothesis)
        metrics["WER"] = output_words.wer
        metrics["MER"] = output_words.mer
        metrics["WIL"] = output_words.wil
        metrics["WIP"] = output_words.wip
        metrics["CER"] = outpur_chars.cer

        return metrics

    def _map_metrics(self, prep_dataset: Dataset) -> Tuple[dict, dict]:
        """
        Map metrics.

        Args:
            prep_dataset: Preprocessed dataset.

        Returns:
            Tuple with erasured metrics and baseline metrics.
        """
        erasured_metrics = {}
        print("Generating original reference...")
        reference = prep_dataset.map(self._get_reference)

        print("Generating original predictions...")
        baseline_preds = prep_dataset.map(self._get_predictions)

        print("Calculating baseline metrics...")
        baseline_metrics = self._calculate_metrics(
            reference["reference"], baseline_preds["transcription"]
        )

        for dim in tqdm(range(self.max_dims), desc="Erasuring Dimensions"):
            dataset_erasured = prep_dataset.map(
                self._apply_erasure, fn_kwargs={"dims": [dim]}
            )
            erasured_preds = dataset_erasured.map(self._get_predictions)
            dim_metrics = self._calculate_metrics(
                reference["reference"], erasured_preds["transcription"]
            )
            erasured_metrics[dim] = dim_metrics

        return erasured_metrics, baseline_metrics

    def _calculate_importance(
        self,
        erasured_metrics: dict,
        baseline_metrics: dict,
    ) -> dict:
        """
        Calculate importance.

        Args:
            erasured_metrics: Erasured metrics.
            baseline_metrics: Baseline metrics.

        Returns:
            Importance.
        """

        importance_dict = {}
        for key, objective in tqdm(
            erasured_metrics.items(), desc="Calculating Importance"
        ):
            importance_dict[key] = {}
            total = 0
            for metric, value in objective.items():
                importance = (
                    -1
                    * (baseline_metrics[metric] - value)
                    / (baseline_metrics[metric] + 1e-8)
                )
                if metric == "WIP":
                    importance = -1 * importance
                importance_dict[key][metric] = importance
                total += importance
            importance_dict[key]["total"] = total

        return importance_dict

    def _custom_scale_importance(self, column: Any) -> Any:
        """
        Custom scale importance.

        Args:
            column: Column to scale.

        Returns:
            Scaled column.
        """

        neg_scaler = MinMaxScaler(feature_range=(-1, 0))
        pos_scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_column = np.zeros_like(column, dtype=float)

        positive_mask = column > 0
        if positive_mask.any():
            pos_scaler.fit(column[positive_mask].values.reshape(-1, 1))
            scaled_column[positive_mask] = pos_scaler.transform(
                column[positive_mask].values.reshape(-1, 1)
            ).flatten()

        negative_mask = column < 0
        if negative_mask.any():
            neg_scaler.fit(column[negative_mask].values.reshape(-1, 1))
            scaled_column[negative_mask] = neg_scaler.transform(
                column[negative_mask].values.reshape(-1, 1)
            ).flatten()

        return scaled_column

    def _apply_custom_scale(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom scale.

        Args:
            dataframe: Dataframe to scale.

        Returns:
            Scaled dataframe.
        """
        print("Scaling importance...")
        scaled_data = pd.DataFrame(
            {
                col: self._custom_scale_importance(dataframe[col])
                for col in dataframe.columns
            }
        )

        return scaled_data

    def _plot_importance(self, importance_dict: dict) -> None:
        """
        Plot importance.

        Args:
            importance_dict: Importance to plot.
        """
        importance_df = pd.DataFrame.from_dict(importance_dict, orient="index")
        scaled_df = self._apply_custom_scale(importance_df)
        imp_heatmap = scaled_df.T
        plt.figure(figsize=(15, 10))
        sns.heatmap(imp_heatmap, annot=False, cmap="coolwarm")
        plt.yticks(rotation=0)
        plt.title("Heatmap de Diferencias Relativas por Dimensión y Métrica")
        plt.xlabel("Dimensión MFCC")
        plt.ylabel("Métrica")
        plt.show()

    def explain(self) -> None:
        """
        Explain model.
        """
        dataset_preprocessed = self._preprocess_dataset(self.dataset)
        erasured_metrics, baseline_metrics = self._map_metrics(dataset_preprocessed)
        importance_dict = self._calculate_importance(erasured_metrics, baseline_metrics)
        self._plot_importance(importance_dict)
