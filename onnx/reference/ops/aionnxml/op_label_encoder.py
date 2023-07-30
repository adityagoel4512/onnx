# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LabelEncoder(OpRunAiOnnxMl):
    def _run(  # type: ignore
        self,
        x,
        default_float=None,
        default_int64=None,
        default_string=None,
        default_as_tensor=None,
        keys_floats=None,
        keys_int64s=None,
        keys_strings=None,
        values_floats=None,
        values_int64s=None,
        values_strings=None,
        keys_as_tensor=None,
        values_as_tensor=None,
    ):
        keys = keys_floats or keys_int64s or keys_strings or keys_as_tensor
        values = values_floats or values_int64s or values_strings or values_as_tensor
        classes = dict(zip(keys, values))

        if values is values_as_tensor:
            defval = default_as_tensor.item()
        elif values is values_floats:
            defval = default_float
        elif values is values_int64s:
            defval = default_int64
        elif values is values_strings:
            defval = default_string
            if not isinstance(defval, str):
                defval = ""
        lookup_func = np.vectorize(lambda x: classes.get(x, defval))
        output = lookup_func(x)
        if output.dtype == object:
            output = output.astype(np.str_)
        return (output,)
