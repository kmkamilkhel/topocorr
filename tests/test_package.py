import re

import topocorr


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|rc\d+)?", topocorr.__version__)
