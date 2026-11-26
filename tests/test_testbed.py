"""Testbed importer: long-format KPM -> 1 s dataset with chronological splits."""
import json

import numpy as np
import pandas as pd

from prep.testbed import import_testbed


def _long_csv(path, seconds=1500, gap=(700, 760)):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(seconds):
        if gap[0] <= i < gap[1]:
            continue
        for k, base in ((1, 50.0), (2, 10.0), (3, 5.0)):
            rows.append({"ts": int((1.7e9 + i + rng.uniform(-0.2, 0.2)) * 1000), "sst": k,
                         "need": base + (i % 7), "used": base})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_import_long_format(tmp_path):
    src = tmp_path / "kpm.csv"
    _long_csv(src)
    mp = tmp_path / "map.yaml"
    mp.write_text("time: {column: ts, unit: ms}\nformat: long\nslice_column: sst\n"
                  "slice_values: {'1': embb, '2': urllc, '3': mmtc}\n"
                  "columns: {req_prb_mean: need, granted_prb: used}\n")
    out, m = import_testbed(str(src), "t", str(mp), out_root=str(tmp_path), chunk_min=3)
    # 60 s gap > max_gap (30 s) -> two pieces; chronological split, train first
    assert sum(len(v) for v in m["splits"].values()) >= 3
    tr = pd.read_parquet(f"{out}/trace_train_000.parquet")
    assert np.allclose(tr["embb_req_prb_mean"].to_numpy()[:14], 50 + np.arange(14) % 7)   # jitter-robust 1 s bins
    assert (tr["urllc_granted_prb"] == 10.0).all()
    assert tr["embb_req_prb_max"].equals(tr["embb_req_prb_mean"])                        # derived column
    starts = [json.load(open(f"{out}/trace_{t}.json"))["start_epoch_s"]
              for sp in ("train", "val", "test") for t in m["splits"][sp]]
    assert starts == sorted(starts)                                                     # no look-ahead
