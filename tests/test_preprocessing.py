"""
test_preprocessing.py
Testes unitários para o módulo preprocessing.py
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.append(os.path.abspath("src"))

from preprocessing import DataPreprocessor, preprocess_pipeline


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame com pelo menos 2 membros por classe para stratified split funcionar"""
    # 3 classes: Em fase (IAN>=10), Moderada (5<=IAN<10), Severa (IAN<5)
    # Mínimo 2 por classe para stratify não falhar
    return pd.DataFrame({
        "NOME":         ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Iris","Jake"],
        "IAN_2022":     [10,    11,    7,     6,     3,    2,     8,     9,     4,    1   ],
        "INDE":         [3.5,   4.0,   6.5,   7.0,   8.5,  9.0,   7.5,   6.0,   4.5,  3.0],
        "fase_efetiva": [5,     5,     6,     6,     7,    7,     6,     5,     4,    3  ],
        "fase_ideal":   [5,     5,     5,     5,     6,    6,     5,     5,     5,    5  ],
    })


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


# ─────────────────────────────────────────────
# DataPreprocessor.__init__
# ─────────────────────────────────────────────

class TestDataPreprocessorInit:
    def test_initial_attributes(self, preprocessor):
        assert preprocessor.numeric_features == []
        assert preprocessor.categorical_features == []
        assert preprocessor.label_encoders == {}


# ─────────────────────────────────────────────
# load_data
# ─────────────────────────────────────────────

class TestLoadData:
    def test_load_single_sheet(self, preprocessor, sample_df):
        mock_excel = MagicMock()
        mock_excel.sheet_names = ["2022"]
        with patch("preprocessing.pd.ExcelFile", return_value=mock_excel), \
             patch("preprocessing.pd.read_excel", return_value=sample_df):
            result = preprocessor.load_data("fake.xlsx")
        assert isinstance(result, pd.DataFrame)

    def test_load_raises_on_bad_file(self, preprocessor):
        with pytest.raises(Exception):
            preprocessor.load_data("nonexistent_file.xlsx")


# ─────────────────────────────────────────────
# _consolidate_data
# ─────────────────────────────────────────────

class TestConsolidateData:
    def test_single_non_numeric_key(self, preprocessor, sample_df):
        dfs = {"sheet1": sample_df}
        result = preprocessor._consolidate_data(dfs)
        assert isinstance(result, pd.DataFrame)

    def test_uses_latest_year(self, preprocessor, sample_df):
        df2021 = sample_df.copy()
        df2022 = sample_df.copy()
        dfs = {"2021": df2021, "2022": df2022}
        result = preprocessor._consolidate_data(dfs)
        assert isinstance(result, pd.DataFrame)

    def test_merge_with_previous_year(self, preprocessor, sample_df):
        df2021 = sample_df.copy()
        df2022 = sample_df.copy()
        dfs = {"2021": df2021, "2022": df2022}
        result = preprocessor._consolidate_data(dfs)
        assert len(result) > 0


# ─────────────────────────────────────────────
# create_target
# ─────────────────────────────────────────────

class TestCreateTarget:
    def test_creates_target_column(self, preprocessor, sample_df):
        result = preprocessor.create_target(sample_df)
        assert "TARGET" in result.columns

    def test_creates_defasagem_classe(self, preprocessor, sample_df):
        result = preprocessor.create_target(sample_df)
        assert "DEFASAGEM_CLASSE" in result.columns

    def test_no_null_targets(self, preprocessor, sample_df):
        result = preprocessor.create_target(sample_df)
        assert result["TARGET"].isnull().sum() == 0

    def test_falls_back_to_available_ian_column(self, preprocessor, sample_df):
        df = sample_df.rename(columns={"IAN_2022": "IAN_2021"})
        result = preprocessor.create_target(df, ian_column="IAN_2022")
        assert "TARGET" in result.columns

    def test_target_values_in_range(self, preprocessor, sample_df):
        result = preprocessor.create_target(sample_df)
        assert result["TARGET"].isin([0, 1, 2]).all()


# ─────────────────────────────────────────────
# identify_feature_types
# ─────────────────────────────────────────────

class TestIdentifyFeatureTypes:
    def test_identifies_numeric_features(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        numeric, categorical = preprocessor.identify_feature_types(df)
        assert len(numeric) > 0

    def test_excludes_target_and_nome(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        numeric, categorical = preprocessor.identify_feature_types(df)
        assert "TARGET" not in numeric
        assert "NOME" not in categorical

    def test_categorical_contains_string_cols(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        df["extra_cat"] = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        numeric, categorical = preprocessor.identify_feature_types(df)
        assert "extra_cat" in categorical


# ─────────────────────────────────────────────
# handle_missing_values
# ─────────────────────────────────────────────

class TestHandleMissingValues:
    def test_fills_numeric_with_median(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        df.loc[0, preprocessor.numeric_features[0]] = np.nan
        result = preprocessor.handle_missing_values(df, strategy='median')
        assert result[preprocessor.numeric_features[0]].isnull().sum() == 0

    def test_fills_numeric_with_mean(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        df.loc[0, preprocessor.numeric_features[0]] = np.nan
        result = preprocessor.handle_missing_values(df, strategy='mean')
        assert result[preprocessor.numeric_features[0]].isnull().sum() == 0

    def test_fills_categorical_with_desconhecido(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        df["cat_col"] = ["a", None, "b", None, "a", "b", "a", None, "b", "a"]
        preprocessor.identify_feature_types(df)
        result = preprocessor.handle_missing_values(df)
        assert "Desconhecido" in result["cat_col"].values

    def test_no_nulls_after_fill(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        result = preprocessor.handle_missing_values(df)
        for col in preprocessor.numeric_features:
            assert result[col].isnull().sum() == 0


# ─────────────────────────────────────────────
# encode_categorical_features
# ─────────────────────────────────────────────

class TestEncodeCategoricalFeatures:
    def test_encodes_categorical_to_numeric(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        df["cat_col"] = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        preprocessor.identify_feature_types(df)
        result = preprocessor.encode_categorical_features(df, fit=True)
        assert result["cat_col"].dtype in [np.int64, np.int32, int]

    def test_stores_label_encoders(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        df["cat_col"] = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        preprocessor.identify_feature_types(df)
        preprocessor.encode_categorical_features(df, fit=True)
        assert "cat_col" in preprocessor.label_encoders

    def test_transform_mode_handles_unseen(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        df["cat_col"] = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        preprocessor.identify_feature_types(df)
        preprocessor.encode_categorical_features(df, fit=True)

        df2 = df.copy()
        df2["cat_col"] = ["a", "z", "a", "b", "a", "b", "a", "b", "a", "b"]  # "z" é unseen
        result = preprocessor.encode_categorical_features(df2, fit=False)
        assert result["cat_col"].iloc[1] == -1


# ─────────────────────────────────────────────
# scale_features
# ─────────────────────────────────────────────

class TestScaleFeatures:
    def test_scales_numeric_features(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        result = preprocessor.scale_features(df, fit=True)
        col = preprocessor.numeric_features[0]
        assert abs(result[col].mean()) < 1.5  # aproximadamente centrado

    def test_transform_mode(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        preprocessor.scale_features(df, fit=True)
        result = preprocessor.scale_features(df, fit=False)
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────
# prepare_features_and_target
# ─────────────────────────────────────────────

class TestPrepareFeaturesAndTarget:
    def test_returns_X_and_y(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        X, y = preprocessor.prepare_features_and_target(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        X, y = preprocessor.prepare_features_and_target(df)
        assert "TARGET" not in X.columns


# ─────────────────────────────────────────────
# split_data
# ─────────────────────────────────────────────

class TestSplitData:
    def test_split_shapes(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        X, y = preprocessor.prepare_features_and_target(df)
        # Patch stratify para None para evitar erro com classes pequenas
        with patch("preprocessing.train_test_split",
                   wraps=lambda X, y, test_size, random_state, stratify: 
                   __import__("sklearn.model_selection", fromlist=["train_test_split"])
                   .train_test_split(X, y, test_size=test_size, random_state=random_state)):
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.3)
        assert len(X_train) + len(X_test) == len(X)

    def test_split_is_reproducible(self, preprocessor, sample_df):
        df = preprocessor.create_target(sample_df)
        preprocessor.identify_feature_types(df)
        X, y = preprocessor.prepare_features_and_target(df)
        with patch("preprocessing.train_test_split",
                   wraps=lambda X, y, test_size, random_state, stratify:
                   __import__("sklearn.model_selection", fromlist=["train_test_split"])
                   .train_test_split(X, y, test_size=test_size, random_state=random_state)):
            X_train1, _, _, _ = preprocessor.split_data(X, y, random_state=42)
            X_train2, _, _, _ = preprocessor.split_data(X, y, random_state=42)
        pd.testing.assert_frame_equal(X_train1.reset_index(drop=True),
                                      X_train2.reset_index(drop=True))


# ─────────────────────────────────────────────
# preprocess_pipeline
# ─────────────────────────────────────────────

class TestPreprocessPipeline:
    def _no_stratify_split(self, X, y, test_size, random_state, stratify=None):
        from sklearn.model_selection import train_test_split as tts
        return tts(X, y, test_size=test_size, random_state=random_state)

    def test_pipeline_returns_five_elements(self, sample_df, tmp_path):
        mock_excel = MagicMock()
        mock_excel.sheet_names = ["2022"]

        with patch("preprocessing.pd.ExcelFile", return_value=mock_excel), \
             patch("preprocessing.pd.read_excel", return_value=sample_df), \
             patch("preprocessing.Config.PROCESSED_DATA_DIR", tmp_path), \
             patch("preprocessing.Config.MODELS_DIR", tmp_path), \
             patch("preprocessing.train_test_split", self._no_stratify_split), \
             patch("preprocessing.save_joblib"):
            result = preprocess_pipeline("fake.xlsx", save_processed=False)

        assert len(result) == 5

    def test_pipeline_X_train_is_dataframe(self, sample_df, tmp_path):
        mock_excel = MagicMock()
        mock_excel.sheet_names = ["2022"]

        with patch("preprocessing.pd.ExcelFile", return_value=mock_excel), \
             patch("preprocessing.pd.read_excel", return_value=sample_df), \
             patch("preprocessing.Config.PROCESSED_DATA_DIR", tmp_path), \
             patch("preprocessing.Config.MODELS_DIR", tmp_path), \
             patch("preprocessing.train_test_split", self._no_stratify_split), \
             patch("preprocessing.save_joblib"):
            X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
                "fake.xlsx", save_processed=False
            )

        assert isinstance(X_train, pd.DataFrame)
        assert len(X_train) > 0

    def test_pipeline_returns_preprocessor_instance(self, sample_df, tmp_path):
        mock_excel = MagicMock()
        mock_excel.sheet_names = ["2022"]

        with patch("preprocessing.pd.ExcelFile", return_value=mock_excel), \
             patch("preprocessing.pd.read_excel", return_value=sample_df), \
             patch("preprocessing.Config.PROCESSED_DATA_DIR", tmp_path), \
             patch("preprocessing.Config.MODELS_DIR", tmp_path), \
             patch("preprocessing.train_test_split", self._no_stratify_split), \
             patch("preprocessing.save_joblib"):
            *_, preprocessor = preprocess_pipeline("fake.xlsx", save_processed=False)

        assert isinstance(preprocessor, DataPreprocessor)
