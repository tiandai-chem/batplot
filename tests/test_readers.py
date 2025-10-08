import os
import numpy as np
from batplot.readers import robust_loadtxt_skipheader


def test_robust_loadtxt_skipheader(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("""# header 1
# header 2
1 2 3
4 5 6
""")
    data = robust_loadtxt_skipheader(str(p))
    assert data.shape == (2, 3)
    assert np.allclose(data[0], [1,2,3])
    assert np.allclose(data[1], [4,5,6])
