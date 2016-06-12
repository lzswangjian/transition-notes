#ifndef EMBEDDING_FEATURE_EXTRACTOR_H_
#define EMBEDDING_FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

/*!
 * \brief An EmbeddingFeatureExtractor manages the extraction of
 * features for embedding-based models. It wraps a sequence of underlying
 * classes of feature extractors, along with associated predicate maps.
 * Each class of feature extractors is associated with a name, e.g., "words",
 * "labels", "tags".
 *
 * The class is split between a generic abstract version,
 * GenericEmbeddingFeatureExtractor (that can be initialized without knowing
 * the signature of the ExtractFeatures method) and a typed version.
 *
 * The predicate maps must be initialized before use: they can be loaded  using
 * Read() or updated via UpdateMapsForExample.
 */
class GenericEmbeddingFeatureExtractor {
  public:
    virtual ~GenericEmbeddingFeatureExtractor() {}

  private:
    // Embedding space names for parameter sharing.
    vector<string> embedding_names_;

    // FML strings for each feature extractor.
    vector<string> embedding_fml_;

    // Size of each of the embedding spaces (maximum predicate id).
    vector<int> embedding_sizes_;

    // Embedding dimensions of the embedding spaces (i.e. 32, 64 etc.)
    vector<int> embedding_dims_;

    // Wheter or not to add string descriptions to converted examples.
    bool add_strings_;
};


class ParserEmbeddingFeatureExtractor
  : public EmbeddingFeatureExtractor<ParserFeatureExtractor, ParserState< {
  public:
    explicit ParserEmbeddingFeatureExtractor(const string &arg_prefix)
      : arg_prefix_(arg_prefix) {}

  private:
    const string ArgPrefix() const override { return arg_prefix_; }

    // Prefix for context parameters.
    string arg_prefix_;
};

#endif
