import importlib
import sys
import types

import pandas as pd


def _import_iowa_utils(monkeypatch):
    sys.modules.pop("LFPAnalysis.iowa_utils", None)

    lfp_preprocess_stub = types.ModuleType("LFPAnalysis.lfp_preprocess_utils")
    lfp_preprocess_stub.load_elec = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "LFPAnalysis.lfp_preprocess_utils", lfp_preprocess_stub)

    return importlib.import_module("LFPAnalysis.iowa_utils")


def test_expand_channel_rows_handles_ranges_and_singletons(monkeypatch):
    iowa_utils = _import_iowa_utils(monkeypatch)

    expanded = iowa_utils._expand_channel_rows(pd.Series(["1", "3:5", "8"]))

    assert expanded == [1, 8, 3, 4, 5]


def test_extract_names_connect_table_groups_channels_by_code(tmp_path, monkeypatch):
    iowa_utils = _import_iowa_utils(monkeypatch)

    table = pd.DataFrame(
        {
            "Code": ["scalp", "CAN", "EKG", "UNUSED", "DepthA"],
            "Contact Location": [
                "Scalp A",
                "Resp belt",
                "EKG",
                "Unused",
                "Left Hipp",
            ],
            "NLX-LFPx channel": ["1:2", "3", "4:5", "6", "7:8"],
        }
    )
    csv_path = tmp_path / "connect.csv"
    table.to_csv(csv_path, index=False)

    eeg_names, resp_names, ekg_names, seeg_names, drop_names = iowa_utils.extract_names_connect_table(
        str(csv_path)
    )

    assert eeg_names == ["lfpx1", "lfpx2"]
    assert resp_names == ["lfpx3"]
    assert ekg_names == ["lfpx4", "lfpx5"]
    assert seeg_names == ["lfpx7", "lfpx8"]
    assert drop_names == ["lfpx6"]
