#include "greedy_parser.h"

GreedyParser::GreedyParser(int batch_size) {
  batch_size_ = batch_size;
  feature_size_ = 3;
  optimizer_ = "ccsgd";
  decay_steps_ = 4000;
  epoch_ = 10;
  learning_rate_ = 0.1;
  weight_decay_ = 1e-4;
  max_grad_norm_ = 5.0;
  step_ = 0;
}

GreedyParser::GreedyParser(int num_actions,
                           vector<int> &num_features,
                           vector<int> &num_feature_ids,
                           vector<int> &embedding_sizes,
                           vector<int> &hidden_layer_sizes) {
  num_actions_ = num_actions;
  num_features_ = num_features;
  num_feature_ids_ = num_feature_ids;
  embedding_sizes_ = embedding_sizes;
  hidden_layer_sizes_ = hidden_layer_sizes;
  feature_size_ = embedding_sizes.size();

  // Set Default Value.
  optimizer_ = "ccsgd";
  decay_steps_ = 4000;
  epoch_ = 10;
  learning_rate_ = 0.1;
  weight_decay_ = 1e-4;
  max_grad_norm_ = 5.0;
  batch_size_ = 1;
  step_ = 0;
}

GreedyParser::~GreedyParser() {
  delete exec_;
  delete opt_;
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
  return Reshape(hidden, Shape(0, num_features * embedding_size));
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
    last_layer = FullyConnected("fc_" + idx_name, last_layer, i2h_weight, i2h_bias, last_layer_size);
    last_layer = Activation("relu_" + idx_name, last_layer, "relu");
    last_layer_size = hidden_layer_sizes_[i];
  }

  // Create Softmax layer.

  auto softmax_weight = Symbol::Variable("softmax_weight");
  auto softmax_bias = Symbol::Variable("softmax_bias");
  auto fc = FullyConnected("softmax_fc", last_layer, softmax_weight, softmax_bias, num_actions_);
  return fc;
}

Symbol GreedyParser::AddCostSymbol() {
  Symbol fc = BuildNetwork();
  Symbol label = Symbol::Variable("label");
  return SoftmaxOutput("softmax", fc, label);
}

void GreedyParser::SetupModel() {
  network_symbol_ = AddCostSymbol();
  Context context_ = Context::cpu();

  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    args_map_[key] = NDArray(Shape(batch_size_, num_features_[i]), context_, false);
  }
  // Infer shape.
  network_symbol_.InferArgsMap(context_, &args_map_, args_map_);
  arg_names_ = network_symbol_.ListArguments();

  for (int i = 0; i < arg_names_.size(); ++i) {
    if (IsParameter(arg_names_[i])) {
      grad_req_type_[arg_names_[i]] = kWriteTo;
      NDArray::SampleUniform(-0.2, 0.2, &args_map_[arg_names_[i]]);
    } else {
      grad_req_type_[arg_names_[i]] = kNullOp;
    }
  }

  exec_ = network_symbol_.SimpleBind(context_, args_map_, map<string, NDArray>(), grad_req_type_);
  opt_ = new Optimizer("ccsgd", learning_rate_, weight_decay_);
  (*opt_).SetParam("momentum", 0.9)
    .SetParam("rescale_grad", 1.0 / batch_size_)
    .SetParam("clip_gradient", max_grad_norm_);

}

// Input features and gold actions.
void GreedyParser::TrainOneBatch(vector<vector<mx_float>> &features,
                                 vector<mx_float> &labels) {
  if (labels.size() != batch_size_) {
    return;
  }
  // Prepare input NDArray.
  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    exec_->arg_dict()[key].SyncCopyFromCPU(features[i]);
    exec_->arg_dict()[key].WaitToRead();
  }
  exec_->arg_dict()["label"].SyncCopyFromCPU(labels);
  exec_->arg_dict()["label"].WaitToRead();

  exec_->Forward(true);
  exec_->Backward();

  acc.Update(exec_->arg_dict()["label"], exec_->outputs[0]);

  // Update parameters.
  for (int i = 0; i < arg_names_.size(); ++i) {
    if (IsParameter(arg_names_[i])) {
      opt_->Update(i, exec_->arg_dict()[arg_names_[i]], exec_->grad_dict()[arg_names_[i]],
                   learning_rate_, weight_decay_);
    }
  }

  step_ += 1;
  if (step_ >= decay_steps_ && step_ % decay_steps_ == 0) {
    learning_rate_ *= pow(decay_rate_, (step_ / decay_steps_));
    LOG(INFO) << "decay learning rate, now lr is [" << learning_rate_
              << "], global step [" << step_ << "]";
  }
}

void GreedyParser::SaveModel(const string &symbol_path, const string &param_path) {
  network_symbol_.Save(symbol_path);
  map<string, NDArray> param_map;
  for (int i = 0; i < arg_names_.size(); ++i) {
    if (IsParameter(arg_names_[i])) {
      param_map[arg_names_[i]] = args_map_[arg_names_[i]];
    }
  }
  NDArray::Save(param_path, param_map);
}

void GreedyParser::LoadModel(const string &symbol_path, const string &param_path) {
  Context context_ = Context::cpu();
  network_symbol_ = Symbol::Load(symbol_path);
  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    args_map_[key] = NDArray(Shape(batch_size_, num_features_[i]), context_, false);
  }

  // Infer shape.
  network_symbol_.InferArgsMap(context_, &args_map_, args_map_);
  map<string, NDArray> param_map = NDArray::LoadToMap(param_path);
  for (auto iter = param_map.begin(); iter != param_map.end(); ++iter) {
    args_map_[iter->first] = iter->second;
  }

  exec_ = network_symbol_.SimpleBind(context_, args_map_);
}

ScoreMatrix GreedyParser::Predict(std::vector<std::vector<mx_float>> &features) {
  // Prepare input NDArray.
  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    exec_->arg_dict()[key].SyncCopyFromCPU(features[i]);
    exec_->arg_dict()[key].WaitToRead();
  }
  exec_->Forward(false);

  ScoreMatrix scoreMatrix;
  vector<NDArray> outputs = exec_->outputs;
  size_t size = outputs[0].Size();
  scoreMatrix.data_ptr_ = new float[size];
  scoreMatrix.row_ = batch_size_;
  scoreMatrix.col_ = size / batch_size_;
  outputs[0].SyncCopyToCPU(scoreMatrix.data_ptr_, size);
  return scoreMatrix;
}
