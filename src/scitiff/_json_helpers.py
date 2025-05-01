# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import re


def _idx_endswith(lines: list[str], *suffix: str) -> int:
    """Finds the first index of a line that ends with any of the given suffixes."""
    for i_l, line in enumerate(lines):
        line = line.strip()
        if any(line.endswith(suf) for suf in suffix):
            return i_l
    raise ValueError(f"Suffix '{suffix}' not found in lines.")


def _join_beautified_array(lines: list[str], cur: str = '') -> str:
    try:
        left_bracket = _idx_endswith(lines, '[')
        right_bracket = _idx_endswith(lines, ']', '],')
        left, array_items, lines = (
            lines[: left_bracket + 1],
            lines[left_bracket + 1 : right_bracket + 1],
            lines[right_bracket + 1 :],
        )
        cur_block = '\n'.join(left)
        array_block = (
            ' '.join([item.strip() for item in array_items])
            .replace(' ]', ']')
            .replace('[ ', '[')
        )
        return _join_beautified_array(lines, cur + '\n' + cur_block + array_block)
    except ValueError:
        return cur + '\n' + '\n'.join(lines) + '\n'


def _json_beautified_array(lines: str) -> str:
    return _join_beautified_array(lines.split('\n'))


def _json_beautified_dict(lines: str) -> str:
    # Regex pattern that matches `{ ... }` that doesn't contain any nested `{ ... }`
    pattern = r'(\{(?:[^{}])*\})'

    matched = re.findall(pattern, lines)
    left = lines[:]
    beutified = ''
    for match in matched:
        idx = left.index(match)
        beutified += left[:idx]
        left = left[idx + len(match) :]
        split_matched = match.split('\n')
        if len(split_matched) == 1:
            beutified += match  # No new line
        else:
            joined = ' '.join([line.strip() for line in split_matched[1:-1]])
            if len(joined) <= 90:
                beutified += '{' + joined + '}'
            else:
                beutified += match
    return beutified + left


def beautify_json(json_str: str) -> str:
    """Beautify the json string."""
    return _json_beautified_dict(_json_beautified_array(json_str))
