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


REGISTER_OP("WordEmbeddingInitializer")
.Output("word_embeddings: float")
.Attr("vectors: string")
.Attr("task_context: string")
.Attr("embedding_init: float = 1.0")
.Doc(R"doc(
Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for
every word specified in a text vocabulary file.

word_embeddings: a tensor containing word embeddings from the specified sstable.
vectors: path to recordio of word embeddings vectors.
task_context: file path at which to read the task context.
)doc");
