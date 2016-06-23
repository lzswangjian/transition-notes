#ifndef EMBEDDING_FEATURE_EXTRACTOR_H_
#define EMBEDDING_FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

#include "feature_extractor.h"
#include "feature_types.h"
#include "parse_features.h"
#include "sentence_features.h"

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

    // Get the prefix string to put in front of all arguments, so they don't
    // conflict with other embedding models.
    virtual const string ArgPrefix() const = 0;

    // Sets up predicate maps and embedding space names that are common for all
    // embedding based feature extractors.
    virtual void Setup(TaskContext *context);
    virtual void Init(TaskContext *context);

    // Requests workspace for the underlying feature extractors. This is
    // implemented in the typed class.
    virtual void RequestWorkspaces(WorkspaceRegistry *registry) = 0;

    // Number of predicates for the embedding at a given index (vocabulary size.)
    int EmbeddingSize(int index) const {
      return generic_feature_extractor(index).GetDomainSize();
    }

    // Returns the number of embedding spaces.
    int NumEmbeddings() const { return embedding_dims_.size(); }

    const int FeatureSize(int idx) const {
      return generic_feature_extractor(idx).feature_types();
    }

    int EmbeddingDims(int index) const { return embedding_dims_[index]; }

    const vector<int> &embedding_dims() const { return embedding_dims_; }

    const vector<string> &embedding_fml() const { return embedding_fml_; }

    string GetParamName(const string &param_name) const {
      return tensorflow::strings::StrCat(ArgPrefix(), "_", param_name);
    }

  protected:
    /*!
     * \brief Provides the generic class with access to the templated extractors.
     * This is used to get the type information out of the feature extractor without
     * knowing the specific calling arguments of the extractor itself.
     */
    virtual const GenericFeatureExtractor &generic_feature_extractor(int idx) const = 0;

    /*!
     * \brief Converts a vector of extracted features into dist_belief::SparseFeatures.
     *
     */
    vector<vector<SparseFeatures>> ConvertExample(
        const vector<FeatureVector> &feature_vectors) const;

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

/*!
 * \brief Templated, obj-specific implemtation of the EmbeddingFeatureExtractor.
 * EXTRACTOR should be a FeatureExtractor<OBJ, ARGS...> class that has the appropriate
 * FeatureTraits() to ensure that locator type features work.
 */
template<class EXTRACTOR, class OBJ, class... ARGS>
class EmbeddingFeatureExtractor : public GenericEmbeddingFeatureExtractor {
  public:
    // Sets up all predicate maps, feature extractors, and flags.
    void Setup(TaskContext *context) override {
      GenericEmbeddingFeatureExtractor::Setup(context);
      feature_extractors_.resize(embedding_fml().size());
      for (int i = 0; i < feature_extractors_.size(); ++i) {
        feature_extractors_[i].Parse(embedding_fml()[i]);
        feature_extractors_[i].Setup(context);
      }
    }

    // Initializes resources needed by the feature extractors.
    void Init(TaskContext *context) override {
      GenericEmbeddingFeatureExtractor::Init(context);
      for (auto &feature_extracor : feature_extractors_) {
        feature_extracor.Init(context);
      }
    }

    // Requests workspaces from the registry. Must be called after Init(),
    // and before Preprocess().
    void RequestWorkspaces(WorkspaceRegistry *registry) override {
      for (auto &feature_extracor : feature_extractors_) {
        feature_extracor.RequestWorkspaces(registry);
      }
    }

    // Must be called on the object one state for each sentence, before any
    // feature extraction (e.g., UpdateMapsForExample, ExtractSparseFeatures).
    void Preprocess(WorkspaceSet *workspaces, OBJ *obj) const {
      for (auto &feature_extracor : feature_extractors_) {
        feature_extractor.Preprocess(workspaces, obj);
      }
    }

    /*!
     * \brief Returns a ragged array of SparseFeatures.
     * for 1) each feature extractor class e, and 2) each feature f extracted by e.
     * Underlying predicate maps will not be updated and so unrecognized predicates may
     * occur. In such a case the SparseFeatures object associated with a given
     * extractor class and feature will be empty.
     */
    vector<vector<SparseFeatures>> ExtractSparseFeatures(
        const WorkspaceSet &workspaces, const OBJ &obj, ARGS... args) const {
      vector<FeatureVector> features(feature_extractors_.size());
      ExtractFeatures(workspaces, obj, args..., &features);
      return ConvertExample(features);
    }

    /*!
     * \brief Extracts features using the extractors.
     * Note that features must already be initialized to the correct number of
     * feature extractors. No predicate mapping is applied.
     */
    void ExtractFeatures(const WorkspaceSet &workspaces, const OBJ &obj,
        ARGS... args, vector<FeatureVector> *features) const {
      DCHECK(features != nullptr);
      DCHECK_EQ(features->size(), feature_extractors_.size());
      for (int i = 0; i < feature_extractors_.size(); ++i){
        (*features)[i].clear();
        feature_extractors_[i].ExtractFeatures(workspaces, obj, args...,
            &(*features)[i]);
      }
    }

  protected:
    // Provides generic access to the feature extractors.
    const GenericFeatureExtractor &generic_feature_extractor(int idx)
      const override {
        DCHECK_LT(idx, feature_extractors_.size());
        DCHECK_GE(idx, 0);
        return feature_extractors_[i];
    }

  private:
    // Templated feature extractor class.
    vector<EXTRACTOR> feature_extractors_;
};

class ParserEmbeddingFeatureExtractor
  : public EmbeddingFeatureExtractor<ParserFeatureExtractor, ParserState> {
  public:
    explicit ParserEmbeddingFeatureExtractor(const string &arg_prefix)
      : arg_prefix_(arg_prefix) {}

  private:
    const string ArgPrefix() const override { return arg_prefix_; }

    // Prefix for context parameters.
    string arg_prefix_;
};

#endif
