/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* StringSplit_doc =
    R"DOC(StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute. The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the delimiter substring. A integer tensor is also returned representing the number of substrings generated. This output tensor has one additional rank compared to the input to store these substrings, as examples below will illustrate.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringSplit,
    20,
    OpSchema()
        .Input(0, "X", "Tensor of strings to split.", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "delimiter",
            "Delimiter to split on. If left unset this defaults to a space character.",
            AttributeProto::STRING,
            false)
        .Attr(
            "maxsplit",
            "Maximum number of splits (from left to right). If left unset, it will make as many splits as possible.",
            AttributeProto::INT,
            false)
        .Output(
            0,
            "Y",
            "Tensor of substrings representing the outcome of splitting the strings in the input on the delimiter. Note that to ensure the same number of elements are present in the final rank, this tensor will pad any necessary empty strings.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            1,
            "Z",
            "The number of substrings generated for each input element.",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T1", {"tensor(string)"}, "The input must be a UTF-8 string tensor")
        .TypeConstraint("T2", {"tensor(string)"}, "Tensor of substrings.")
        .TypeConstraint("T3", {"tensor(int32)"}, "The number of substrings generated.")
        .SetDoc(StringSplit_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          const TypeProto* input_type = ctx.getInputType(0);
          if (input_type == nullptr || !input_type->has_tensor_type() ||
              input_type->tensor_type().elem_type() != TensorProto::STRING) {
            return;
          }

          // We produce a string tensor per input element. Therefore we have one additional rank with a runtime
          // dependent number of elements. The result of the output shape can be inferred directly from the input
          // however.
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
          getOutputShape(ctx, 0)->add_dim();

          // The output tensor containing the number of substrings has identical shape to the input but produces int32
          // results.
          ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::INT32);
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }));

static const char* StringNormalizer_ver10_doc = R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringNormalizer,
    10,
    OpSchema()
        .Input(0, "X", "UTF-8 strings to normalize", "tensor(string)")
        .Output(0, "Y", "UTF-8 Normalized strings", "tensor(string)")
        .Attr(
            std::string("case_change_action"),
            std::string("string enum that cases output to be lowercased/uppercases/unchanged."
                        " Valid values are \"LOWER\", \"UPPER\", \"NONE\". Default is \"NONE\""),
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            std::string("is_case_sensitive"),
            std::string("Boolean. Whether the identification of stop words in X is case-sensitive. Default is false"),
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "stopwords",
            "List of stop words. If not set, no word would be removed from X.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "locale",
            "Environment dependent string that denotes the locale according to which output strings needs to be upper/lowercased."
            "Default en_US or platform specific equivalent as decided by the implementation.",
            AttributeProto::STRING,
            OPTIONAL_VALUE)
        .SetDoc(StringNormalizer_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::STRING);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          TensorShapeProto output_shape;
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto dim_size = input_shape.dim_size();
          // Last axis dimension is unknown if we have stop-words since we do
          // not know how many stop-words are dropped
          if (dim_size == 1) {
            // Unknown output dimension
            output_shape.add_dim();
          } else if (dim_size == 2) {
            // Copy B-dim
            auto& b_dim = input_shape.dim(0);
            if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
              fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
            }
            *output_shape.add_dim() = b_dim;
            output_shape.add_dim();
          } else {
            fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
          }
          updateOutputShape(ctx, 0, output_shape);
        }));

} // namespace ONNX_NAMESPACE
