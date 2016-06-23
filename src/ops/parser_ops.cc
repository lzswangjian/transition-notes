#include "tensorflow/core/framework/op.h"

REGISTER_OP("GoldParseReader")
  .Output("features: feature_size * string")
  .Output("num_epochs: int32")
  .Output("gold_actions: int32")
  .Attr("task_context: string")
  .Attr("feature_size: int")
  .Attr("batch_size: int")
  .Attr("corpus_name: string='documents'")
  .Attr("arg_prefix: string='brain_parser'")
  .SetIsStateful();
