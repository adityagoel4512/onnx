# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,W0221

from typing import Union

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


def pad_empty_string(
    split_lists: Union[list, np.ndarray], padding_requirement: Union[list, int]
):
    # pylint: disable=unidiomatic-typecheck`
    if type(split_lists) is list:
        return split_lists + ["" for _ in range(padding_requirement)]
    elif type(split_lists) is np.ndarray:
        return list(map(pad_empty_string, split_lists, padding_requirement))
    else:
        raise TypeError("Invalid array type")


def split_with_padding(x, separator=" ", maxsplit=None):
    split_lists = np.char.split(x.astype(np.str_), separator, maxsplit)
    # Find the maximum length after splitting
    num_splits = np.vectorize(len)(split_lists).astype(np.int32)
    padding_requirement = (np.max(num_splits) - num_splits).tolist()
    split_lists_padded = np.array(
        pad_empty_string(split_lists, padding_requirement), dtype=object
    )
    # Add padding to lists that are shorter than the maximum length
    return split_lists_padded, num_splits


class StringSplit(OpRun):
    def _run(self, x, delimiter=None, maxsplit=None):
        if delimiter is None:
            delimiter = " "

        if x.dtype.kind not in _acceptable_str_dtypes:
            raise TypeError(f"Inputs must be string tensors, received dtype {x.dtype}")
        return split_with_padding(x, delimiter, maxsplit)