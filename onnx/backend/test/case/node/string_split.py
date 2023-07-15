# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class StringSplit(Base):
    @staticmethod
    def export_basic() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            delimiter=".",
            maxsplit=None,
        )

        x = np.array(["abc.com", "def.net"]).astype(object)

        substrings = np.array([["abc", "com"], ["def", "net"]]).astype(object)

        length = np.array([2, 2], dtype=np.int32)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_basic",
        )

    @staticmethod
    def export_maxsplit() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["substrings", "length"],
            maxsplit=2,
        )

        x = np.array(
            [["hello world", "def.net"], ["o n n x", "the quick brown fox"]]
        ).astype(object)

        substrings = np.array(
            [
                [["hello", "world", ""], ["def.net", "", ""]],
                [["o", "n", "n x"], ["the", "quick", "brown fox"]],
            ]
        ).astype(object)

        length = np.array([[2, 1], [3, 3]], np.int32)

        expect(
            node,
            inputs=[x],
            outputs=[substrings, length],
            name="test_string_split_maxsplit",
        )
