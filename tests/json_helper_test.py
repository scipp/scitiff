# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json

from scitiff._json_helpers import beautify_json


def test_beautifier_does_not_delete_anything():
    """Test that the beautifier does not delete anything."""

    example = {
        "a": 1,
        "b": 2,
        "c": [1, 2, 3],
        "d": {"e": 1, "f": 2, "g": [1, 2, 3]},
        "e": [
            {"h": 1, "i": 2, "j": [1, 2, 3], "k": {"l": "SomeVeryLongString" * 50}},
        ],
    }
    json_str = json.dumps(example, indent=2)
    beautified_json = beautify_json(json_str)
    beautified_dict = json.loads(beautified_json)
    assert example == beautified_dict
