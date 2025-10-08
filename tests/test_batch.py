import os
import numpy as np
from batplot.batch import batch_process

class Args:
    def __init__(self):
        self.xaxis = 'Q'  # ensure .txt uses a concrete axis
        self.wl = None
        self.raw = True


def test_batch_exports_svg(tmp_path, monkeypatch):
    # Create two simple files
    f1 = tmp_path / "a.txt"
    f1.write_text("""# head\n0 0\n1 1\n""")
    f2 = tmp_path / "b.qye"
    f2.write_text("""0 0\n1 1\n""")

    # Auto-overwrite without prompting
    from batplot import utils
    monkeypatch.setattr(utils, "_confirm_overwrite", lambda p, auto_suffix=True: p)

    args = Args()
    batch_process(str(tmp_path), args)

    out_dir = tmp_path / "batplot_svg"
    assert out_dir.exists()
    svgs = list(out_dir.glob("*.svg"))
    assert len(svgs) >= 1
