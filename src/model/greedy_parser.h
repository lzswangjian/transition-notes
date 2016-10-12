#ifndef GREEDY_PARSER_H_
#define GREEDY_PARSER_H_

#include <mxnet-cpp/MxNetCpp.h>

#include "../utils/utils.h"

using namespace std;
using namespace mxnet::cpp;

// Builds a Chen & Manning style greedy neural net parser.
//
// Args:
//  num_actions: int size of the set of parser actions.
//  num_features: int list of dimensions of the feature vectors.
//  num_feature_ids: int list of same length as num_features corresponding 
//                   to the sizes of the input feature spaces.
//  embedding_sizes: int list of same length as num_features of the 
//                   desired embedding layer sizes.
class GreedyParser {
public:
    GreedyParser(int num_actions,
            vector<int> &num_features,
            vector<int> &num_feature_ids,
            vector<int> &embedding_sizes,
            vector<int> &hidden_layer_sizes);

    Symbol AddEmbedding(mx_uint num_features,
                mx_uint vocab_size,
                mx_uint embedding_size,
                const string &name);

  inline mx_uint GetEmbeddingSize() { 
      mx_uint size = 0;
      for (int i = 0; i < feature_size_; ++i) {
          size += num_features_[i] * embedding_sizes_[i];
      }
      return size;
  }

  Symbol BuildNetwork();

  inline bool IsParamerter(const string &name) {
    return false;
  }

  void SetupModel();

public:
  void TrainModel(std::vector<std::vector<mx_float>> &features,
                  std::vector<mx_float> &labels);


private:
  int num_actions_;
  vector<int> num_features_;
  vector<int> num_feature_ids_;
  vector<int> embedding_sizes_;
  vector<int> hidden_layer_sizes_;
  int feature_size_;
  
  mx_float learning_rate_;
  mx_float max_grad_norm_;
  string optimizer_;
  mx_uint epoch_;
  mx_uint batch_size_;
  mx_uint decay_steps_;
  mx_float decay_rate_;
  mx_float weight_decay_;

  Symbol network_symbol_;
  Optimizer *opt_;
  std::map<std::string, NDArray> args_map_;
  std::map<std::string, NDArray> aux_map_;
};

#endif /* end of include guard: GREEDY_PARSER_H_ */
