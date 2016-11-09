#ifndef GREEDY_PARSER_H_
#define GREEDY_PARSER_H_

#include <mxnet-cpp/mxnet_cpp.h>

#include "../utils/utils.h"
#include "score_matrix.h"

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
  GreedyParser(int batch_size);

  GreedyParser(int num_actions,
               vector<int> &num_features,
               vector<int> &num_feature_ids,
               vector<int> &embedding_sizes,
               vector<int> &hidden_layer_sizes);

  virtual ~GreedyParser();

 public:

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

  Symbol AddCostSymbol();

  inline bool IsParameter(const string &name) {
    return utils::StringEndWith(name, "weight") || utils::StringEndWith(name, "bias");
  }

  virtual void SetupModel();

 public:
  void TrainOneBatch(std::vector<std::vector<mx_float>> &features,
                     std::vector<mx_float> &labels);

  ScoreMatrix Predict(std::vector<std::vector<mx_float>> &features);

  void SaveModel(const std::string &symbol_path, const std::string &param_path);

  void LoadModel(const std::string &symbol_path, const std::string &param_path);

 protected:
  int num_actions_;
  vector<int> num_features_;
  vector<int> num_feature_ids_;
  vector<int> embedding_sizes_;
  vector<int> hidden_layer_sizes_;
  int feature_size_;

  int step_;

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

  vector<string> arg_names_;

  map<string, NDArray> args_map_;
  map<string, OpReqType> grad_req_type_;
  Executor *exec_;

 public:
  Accuracy acc;
};

#endif /* end of include guard: GREEDY_PARSER_H_ */
