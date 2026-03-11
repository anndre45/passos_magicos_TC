"""
test_utils.py
Testes unitários para o módulo utils.py
"""

import sys
import os
import pytest
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath("src"))

from utils import (
    Config,
    save_pickle,
    load_pickle,
    save_joblib,
    load_joblib,
    save_json,
    load_json,
    classify_defasagem,
    classify_pedra,
    get_timestamp,
    log_dataframe_info,
    calculate_ian_from_phases,
    encode_defasagem_target,
    decode_defasagem_target,
    get_memory_usage,
    reduce_memory_usage,
)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class TestConfig:
    def test_base_dir_is_path(self):
        assert isinstance(Config.BASE_DIR, Path)

    def test_test_size_between_0_and_1(self):
        assert 0 < Config.TEST_SIZE < 1

    def test_random_state_is_int(self):
        assert isinstance(Config.RANDOM_STATE, int)

    def test_cv_folds_positive(self):
        assert Config.CV_FOLDS > 0

    def test_defasagem_classes_keys(self):
        assert set(Config.DEFASAGEM_CLASSES.keys()) == {'Em fase', 'Moderada', 'Severa'}

    def test_pedra_ranges_has_four_entries(self):
        assert len(Config.PEDRA_RANGES) == 4

    def test_create_directories(self, tmp_path):
        with patch.object(Config, 'BASE_DIR', tmp_path):
            Config.DATA_DIR = tmp_path / 'data'
            Config.RAW_DATA_DIR = Config.DATA_DIR / 'raw'
            Config.PROCESSED_DATA_DIR = Config.DATA_DIR / 'processed'
            Config.MODELS_DIR = tmp_path / 'models'
            Config.LOGS_DIR = tmp_path / 'logs'
            Config.create_directories()
            assert Config.MODELS_DIR.exists()
            assert Config.LOGS_DIR.exists()


# ─────────────────────────────────────────────
# save_pickle / load_pickle
# ─────────────────────────────────────────────

class TestPickle:
    def test_save_and_load(self, tmp_path):
        obj = {"key": [1, 2, 3]}
        path = tmp_path / "test.pkl"
        save_pickle(obj, path)
        loaded = load_pickle(path)
        assert loaded == obj

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            load_pickle(tmp_path / "nonexistent.pkl")

    def test_save_raises_on_bad_path(self):
        with pytest.raises(Exception):
            save_pickle({}, "/invalid_dir/file.pkl")


# ─────────────────────────────────────────────
# save_joblib / load_joblib
# ─────────────────────────────────────────────

class TestJoblib:
    def test_save_and_load(self, tmp_path):
        obj = [10, 20, 30]
        path = tmp_path / "model.joblib"
        save_joblib(obj, path)
        loaded = load_joblib(path)
        assert loaded == obj

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            load_joblib(tmp_path / "missing.joblib")


# ─────────────────────────────────────────────
# save_json / load_json
# ─────────────────────────────────────────────

class TestJson:
    def test_save_and_load(self, tmp_path):
        data = {"model": "RF", "score": 0.95}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            load_json(tmp_path / "missing.json")

    def test_json_content_is_valid(self, tmp_path):
        data = {"a": 1}
        path = tmp_path / "valid.json"
        save_json(data, path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["a"] == 1


# ─────────────────────────────────────────────
# classify_defasagem
# ─────────────────────────────────────────────

class TestClassifyDefasagem:
    def test_em_fase(self):
        assert classify_defasagem(10) == 'Em fase'

    def test_em_fase_above_10(self):
        assert classify_defasagem(15) == 'Em fase'

    def test_moderada_exact_5(self):
        assert classify_defasagem(5) == 'Moderada'

    def test_moderada_between_5_and_10(self):
        assert classify_defasagem(7) == 'Moderada'

    def test_severa(self):
        assert classify_defasagem(3) == 'Severa'

    def test_severa_zero(self):
        assert classify_defasagem(0) == 'Severa'

    def test_nan_returns_none(self):
        assert classify_defasagem(float('nan')) is None

    def test_nan_with_pd_na(self):
        assert classify_defasagem(pd.NA) is None


# ─────────────────────────────────────────────
# classify_pedra
# ─────────────────────────────────────────────

class TestClassifyPedra:
    def test_quartzo(self):
        assert classify_pedra(4.0) == 'Quartzo'

    def test_agata(self):
        assert classify_pedra(6.5) == 'Ágata'

    def test_ametista(self):
        assert classify_pedra(7.5) == 'Ametista'

    def test_topazio(self):
        assert classify_pedra(8.5) == 'Topázio'

    def test_nan_returns_none(self):
        assert classify_pedra(float('nan')) is None

    def test_out_of_range_returns_quartzo(self):
        assert classify_pedra(0.5) == 'Quartzo'


# ─────────────────────────────────────────────
# get_timestamp
# ─────────────────────────────────────────────

class TestGetTimestamp:
    def test_returns_string(self):
        ts = get_timestamp()
        assert isinstance(ts, str)

    def test_format_length(self):
        ts = get_timestamp()
        assert len(ts) == 15  # YYYYmmdd_HHMMSS


# ─────────────────────────────────────────────
# log_dataframe_info
# ─────────────────────────────────────────────

class TestLogDataframeInfo:
    def test_runs_without_error(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        log_dataframe_info(df, "TestDF")  # deve executar sem exceção


# ─────────────────────────────────────────────
# calculate_ian_from_phases
# ─────────────────────────────────────────────

class TestCalculateIanFromPhases:
    def test_on_track_returns_10(self):
        assert calculate_ian_from_phases(5, 5) == 10

    def test_ahead_returns_10(self):
        assert calculate_ian_from_phases(6, 5) == 10

    def test_one_behind_returns_5(self):
        assert calculate_ian_from_phases(4, 5) == 5

    def test_two_or_more_behind_returns_2_5(self):
        assert calculate_ian_from_phases(2, 5) == 2.5

    def test_invalid_input_returns_none(self):
        assert calculate_ian_from_phases("abc", 5) is None


# ─────────────────────────────────────────────
# encode_defasagem_target / decode_defasagem_target
# ─────────────────────────────────────────────

class TestEncodeDecodeDefasagem:
    def test_encode_em_fase(self):
        s = pd.Series(['Em fase'])
        assert encode_defasagem_target(s).iloc[0] == 0

    def test_encode_moderada(self):
        s = pd.Series(['Moderada'])
        assert encode_defasagem_target(s).iloc[0] == 1

    def test_encode_severa(self):
        s = pd.Series(['Severa'])
        assert encode_defasagem_target(s).iloc[0] == 2

    def test_decode_reverses_encode(self):
        original = pd.Series(['Em fase', 'Moderada', 'Severa'])
        encoded = encode_defasagem_target(original)
        decoded = decode_defasagem_target(encoded)
        pd.testing.assert_series_equal(original, decoded)

    def test_encode_all_three_classes(self):
        s = pd.Series(['Em fase', 'Moderada', 'Severa'])
        result = encode_defasagem_target(s)
        assert list(result) == [0, 1, 2]


# ─────────────────────────────────────────────
# get_memory_usage
# ─────────────────────────────────────────────

class TestGetMemoryUsage:
    def test_returns_string_with_mb(self):
        df = pd.DataFrame({"a": range(100)})
        result = get_memory_usage(df)
        assert "MB" in result

    def test_larger_df_uses_more_memory(self):
        df_small = pd.DataFrame({"a": range(10)})
        df_large = pd.DataFrame({"a": range(100000)})
        small_mem = float(get_memory_usage(df_small).replace(" MB", ""))
        large_mem = float(get_memory_usage(df_large).replace(" MB", ""))
        assert large_mem > small_mem


# ─────────────────────────────────────────────
# reduce_memory_usage
# ─────────────────────────────────────────────

class TestReduceMemoryUsage:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3]
        })
        result = reduce_memory_usage(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_preserved(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3]
        })
        result = reduce_memory_usage(df)
        assert list(result.columns) == list(df.columns)

    def test_values_preserved(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = reduce_memory_usage(df)
        assert list(result["a"]) == [1, 2, 3]
