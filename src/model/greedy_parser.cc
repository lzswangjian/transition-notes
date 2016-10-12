#include "greedy_parser.h"

GreedyParser::GreedyParser(int num_actions,
            vector<int> &num_features,
            vector<int> &num_feature_ids,
            vector<int> &embedding_sizes,
            vector<int> &hidden_layer_sizes) {
    num_actions_ = num_actions;
    num_features_ = num_features;
    num_feature_ids_ = num_feature_ids;
    embedding_sizes_ = embedding_sizes;
    feature_size_ = embedding_sizes.size();

    // Set Default Value.
    optimizer_ = "sgd";
    decay_steps_ = 4000;
    epoch_ = 10;
    learning_rate_ = 0.1;
    weight_decay_ = 1e-4;
    max_grad_norm_ = 5.0;
    batch_size_  = 32;
}


Symbol GreedyParser::AddEmbedding(mx_uint num_features,
                                  mx_uint vocab_size,
                                  mx_uint embedding_size,
                                  const string &name) {
    string data_name = "feature_" + name + "_data";
    auto data = Symbol::Variable(data_name);
    string embed_weight_name = name + "_embed_weight";
    auto embedding_weight = Symbol::Variable(embed_weight_name);
    auto hidden = Embedding("embedding_" + name, data, embedding_weight,
            vocab_size, embedding_size);
    return Reshape("reshape_" + name, hidden, Shape(0, num_features * embedding_size));
}

Symbol GreedyParser::BuildNetwork() {
  // Creating embedding layer.
  vector<Symbol> embeddings;
  for (mx_uint i = 0; i < feature_size_; ++i) {
    embeddings.push_back(AddEmbedding(num_features_[i],
                                      num_feature_ids_[i],
                                      embedding_sizes_[i],
                                      utils::Printf(i)));
  }
  
  auto last_layer = Concat(embeddings, 3, 1);
  mx_uint last_layer_size = GetEmbeddingSize();

  // Create ReLU layers.
  for (mx_uint i = 0; i < hidden_layer_sizes_.size(); ++i) {
    string idx_name = utils::Printf(i);
    auto i2h_weight = Symbol::Variable(idx_name + "_i2h_weight");
    auto i2h_bias = Symbol::Variable(idx_name + "_i2h_bias");
    last_layer = FullyConnected("fc_"+idx_name, last_layer, i2h_weight, i2h_bias, last_layer_size);
    last_layer = Activation("relu_"+idx_name, last_layer, "relu");
    last_layer_size = hidden_layer_sizes_[i];
  }

  // Create Softmax layer.
  auto label = Symbol::Variable("label");
  auto softmax_weight = Symbol::Variable("softmax_weight");
  auto softmax_bias = Symbol::Variable("softmax_bias");
  auto fc  = FullyConnected("softmax_fc", last_layer, softmax_weight, softmax_bias, num_actions_);
  return SoftmaxOutput("softmax", fc, label);
}

void GreedyParser::SetupModel() {
  network_symbol_ = BuildNetwork();
  Context context_ = Context::cpu();

  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    args_map_[key] = NDArray(Shape(batch_size_, num_features_[i]), context_);
  }

  network_symbol_.InferArgsMap(context_, &args_map_, args_map_);

  Optimizer *opt = new Optimizer("sgd", learning_rate_, weight_decay_);
  (*opt).SetParam("momentum", 0.9)
    .SetParam("clip_gradient", max_grad_norm_);
  
}

// Input features and gold actions.
void GreedyParser::TrainModel(vector<vector<mx_float>> &features,
                              vector<mx_float> &labels) {
  // Prepare input NDArray.
  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    args_map_[key].SyncCopyFromCPU(features[i]);
  }
  args_map_["label"].SyncCopyFromCPU(labels);
  NDArray::WaitAll();

  // Bind to exec.
  Context context_ = Context::cpu();
  auto *exec = network_symbol_.SimpleBind(context_, args_map_);
  exec->Forward(true);
  exec->Backward();
  exec->UpdateAll(opt_, learning_rate_, weight_decay_);
  delete exec;
}

