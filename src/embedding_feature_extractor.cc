#include "embedding_feature_extractor.h"

void GenericEmbeddingFeatureExtractor::Setup(TaskContext *context) {
}

void GenericEmbeddingFeatureExtractor::Init(TaskContext *context) {
}

vector<vector<SparseFeatures>> GenericEmbeddingFeatureExtractor::ConvertExample(
    const vector<FeatureVector> &feature_vectors) const {
  // Extract the features.
  vector<vector<SparseFeatures>> sparse_features(feature_vectors.size());
  // Detail
  
  return sparse_features;
}
