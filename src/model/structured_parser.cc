#include "structured_parser.h"

StructuredParser::StructuredParser(int batch_size)
  : GreedyParser(batch_size), compute_context_(DeviceType::kCPU, 0) {
  // Do Initialization
  max_steps_ = 25;
  beam_size_ = 4;

}

StructuredParser::StructuredParser(int batch_size,
                                   int num_actions,
                                   vector<int> &num_features,
                                   vector<int> &num_feature_ids,
                                   vector<int> &embedding_sizes,
                                   vector<int> &hidden_layer_sizes)
: GreedyParser(num_actions, num_features, num_feature_ids, embedding_sizes, hidden_layer_sizes),
  compute_context_(DeviceType::kCPU, 0) {
  batch_size_ =batch_size;
  max_steps_ = 25;
  beam_size_ = 4;
  epoch_ = 4;
  scoreMatrixDptr = new float[batch_size_ * beam_size_ * num_actions_];

}

StructuredParser::~StructuredParser() {
  if (scoreMatrixDptr != nullptr) {
    delete[] scoreMatrixDptr;
  }
  for (size_t i = 0; i < exec_list_.size(); ++i) {
    if (exec_list_[i] != nullptr) {
      delete exec_list_[i];
    }
  }
}


void StructuredParser::CreateOptimizer(const string &opt) {
    opt_ = new Optimizer("ccsgd", learning_rate_, weight_decay_);
  (*opt_).SetParam("momentum", 0.9)
    .SetParam("rescale_grad", 1.0 / batch_size_)
    .SetParam("clip_gradient", max_grad_norm_);
}

void StructuredParser::Padding(vector<vector<float>> &feature_outputs) {
  int num_of_states = (int) (feature_outputs[0].size() / num_features_[0]);
  // LOG(INFO) << "Padding: BatchState has [" << num_of_states << "]";
  if (num_of_states == batch_size_ * beam_size_) {
    return;
  }

  // padding.
  for (int pad_size = num_of_states; pad_size < batch_size_ * beam_size_; ++pad_size) {
    for (size_t k = 0; k < feature_size_; ++k) {
      for (int fsize = 0; fsize < num_features_[k]; ++fsize) {
        feature_outputs[k].push_back(0);
      }
    }
  }
}

int StructuredParser::TrainIter() {
  int accumulate_steps = 0;
  bool all_alive = true;

  // Forward
  int epoch;
  vector<vector<float>> feature_outputs;
  beam_reader_->Compute(context, feature_outputs, epoch);
  if (epoch >= epoch_) {
      return epoch;
  }
  Padding(feature_outputs);
  while (accumulate_steps < max_steps_ && all_alive) {
    // Prepare Batch Data.
    for (mx_uint i = 0; i < feature_size_; ++i) {
        string key = "feature_" + utils::Printf(i) + "_data";
        exec_list_[accumulate_steps]->arg_dict()[key].SyncCopyFromCPU(feature_outputs[i]);
        exec_list_[accumulate_steps]->arg_dict()[key].WaitToRead();
    }
    // Do forward prediction.
    exec_list_[accumulate_steps]->Forward(true);

    vector<NDArray> outputs = exec_list_[accumulate_steps]->outputs;
    // Convert output into score_matrix.
    ScoreMatrix scoreMatrix;
    size_t size = outputs[0].Size();
    scoreMatrix.data_ptr_ = scoreMatrixDptr;  // Change to Fixed Array.
    scoreMatrix.row_ = batch_size_ * beam_size_;
    scoreMatrix.col_ = num_actions_;
    outputs[0].SyncCopyToCPU(scoreMatrix.data_ptr_, size);
    // PopulateFeatureOutputs & Get All Beam States.
    feature_outputs.clear();
    beam_parser_->Compute(context, beam_reader_->batch_state_.get(),
                          scoreMatrix, feature_outputs, all_alive);
    Padding(feature_outputs);

    accumulate_steps += 1;
  }
  LOG(INFO) << "accumulate_steps: [" << accumulate_steps << "]";

  // Make Cross Entropy Loss and Compute Grads.
  beam_parser_output_->Compute(context, beam_reader_->batch_state_.get());
  vector<NDArray> step_head_grad;
  CrossEntropy(step_head_grad, accumulate_steps);

  // Backward
  for (size_t eidx = 0; eidx < exec_list_.size() && eidx < accumulate_steps; ++eidx) {
    vector<NDArray> head_grad{step_head_grad[eidx]};
    exec_list_[eidx]->Backward(head_grad);
    // Update parameters.
    for (int i = 0; i < arg_names_.size(); ++i) {
      if (IsParameter(arg_names_[i])) {
        opt_->Update(i, exec_list_[eidx]->arg_dict()[arg_names_[i]],
                     exec_list_[eidx]->grad_dict()[arg_names_[i]],
                     learning_rate_, weight_decay_);
      }
    }
  }

  return epoch;
}

void StructuredParser::CrossEntropy(vector<NDArray> &step_head_ndarray,
                                    int accumulate_steps) {
  // Get Beam Path Scores.
  vector<int32_t> beam_ids = beam_parser_output_->beam_ids_;
  // vector<int32_t> slot_ids = beam_parser_output_->slot_ids_;
  vector<int32_t> beam_step_sizes = beam_parser_output_->beam_step_sizes_;
  // batch_size
  vector<int32_t> gold_slot = beam_parser_output_->gold_slot_;
  vector<float> path_scores = beam_parser_output_->path_scores_;
  vector<int32_t> indices = beam_parser_output_->indices_;

  int start = 0;
  vector<vector<float> > step_head_grads(accumulate_steps);
  for (int i = 0; i < accumulate_steps; ++i) {
    step_head_grads[i].resize(batch_size_ * beam_size_ * num_actions_);
    for (int j = 0; j < step_head_grads[i].size(); ++j) {
      step_head_grads[i][j] = 0.0f;
    }
  }

  for (int beam_id = 0; beam_id < batch_size_; ++beam_id) {
    // if (beam_reader_->batch_state_.get()->Beam(beam_id).gold_ == nullptr) continue;
    vector<float> energy;
    for (int j = start; j < path_scores.size(); ++j) {
      if (beam_ids[j] == beam_id) {
        energy.push_back(path_scores[j]);
      } else {
        start = j;
        break;
      }
    }
    vector<float> softmax;
    vector<float> softmax_grad;
    if (softmax.size() != beam_size_) continue;
    Softmax(energy, softmax);
    SoftmaxGrad(softmax, softmax_grad, gold_slot[beam_id]);

    int beam_step_size = beam_step_sizes[beam_id];

    // According Beam Search Path to BackPropagate Gradient.
    for (size_t gidx = 0; gidx < softmax_grad.size(); ++gidx) {
      for (size_t step = 0; step < beam_step_size; ++step) {
        int kidx = beam_id * gidx * beam_step_size;
        step_head_grads[step][indices[kidx + step]] += softmax_grad[gidx];
      }
    }
  }

  // Create Head Grad NDArray.
  for (int i = 0; i < accumulate_steps; ++i) {
    NDArray t(Shape(batch_size_ * beam_size_, num_actions_), compute_context_, false);
    t.SyncCopyFromCPU(step_head_grads[i].data(), batch_size_ * beam_size_ * num_actions_);
    step_head_ndarray.push_back(t);
  }
}

void StructuredParser::SetupModel(Symbol symbol) {
  Context context_ = Context::cpu();

  for (mx_uint i = 0; i < feature_size_; ++i) {
    string key = "feature_" + utils::Printf(i) + "_data";
    args_map_[key] = NDArray(Shape(batch_size_ * beam_size_, num_features_[i]), context_, false);
  }

  // Infer shape.
  symbol.InferArgsMap(context_, &args_map_, args_map_);
  arg_names_ = symbol.ListArguments();

  for (int i = 0; i < arg_names_.size(); ++i) {
    if (IsParameter(arg_names_[i])) {
      grad_req_type_[arg_names_[i]] = kWriteTo;
    } else {
      grad_req_type_[arg_names_[i]] = kNullOp;
    }
  }
}

void StructuredParser::InitWithPreTrainedParameters(const string &param_path) {
  map<string, NDArray> param_map = NDArray::LoadToMap(param_path);
  for (auto iter = param_map.begin(); iter != param_map.end(); ++iter) {
    args_map_[iter->first] = iter->second;
  }
}

void StructuredParser::InitFreshParameters() {
  for (int i = 0; i < arg_names_.size(); ++i) {
      if (IsParameter(arg_names_[i])) {
          NDArray::SampleUniform(-0.2, 0.2, &args_map_[arg_names_[i]]);
      }
  }
}

void StructuredParser::BuildSequence() {
    // Genreate `max_steps_` greedy network.
  Context context_ = Context::cpu();
  bool has_inferred_args = false;
  exec_list_.resize(max_steps_);
  for (size_t i = 0; i < max_steps_; ++i) {
      Symbol s = BuildNetwork();
      if (!has_inferred_args) {
          // Only Setup Model Once (Parameters Shared).
          SetupModel(s);
          has_inferred_args = true;
      }
      Executor *exec = s.SimpleBind(context_, args_map_,
              map<string, NDArray>(), grad_req_type_);
      exec_list_[i] = exec;
  }
}

void StructuredParser::SaveModel(const std::string &param_path) {
  map<string, NDArray> param_map;
  for (int i = 0; i < arg_names_.size(); ++i) {
    if (IsParameter(arg_names_[i])) {
      param_map[arg_names_[i]] = args_map_[arg_names_[i]];
    }
  }
  NDArray::Save(param_path, param_map);
}

void StructuredParser::ConfigEvalModel(const string &param_path) {
  Context context_ = Context::cpu();
  Symbol s = BuildNetwork();
  SetupModel(s);

  map<string, NDArray> param_map = NDArray::LoadToMap(param_path);
  for (auto iter = param_map.begin(); iter != param_map.end(); ++iter) {
    args_map_[iter->first] = iter->second;
  }

  exec_ = s.SimpleBind(context_, args_map_,
                                map<string, NDArray>(), grad_req_type_);
}

int StructuredParser::PredictOneBatch(int max_steps, vector<string> &documents) {
  int accumulate_steps = 0;
  bool all_alive = true;

  // Forward
  int epoch;
  vector<vector<float>> feature_outputs;
  beam_reader_->Compute(context, feature_outputs, epoch);
  if (epoch >= epoch_) {
    return epoch;
  }
  Padding(feature_outputs);
  while (accumulate_steps < max_steps && all_alive) {
    // Prepare Batch Data.
    for (mx_uint i = 0; i < feature_size_; ++i) {
      string key = "feature_" + utils::Printf(i) + "_data";
      exec_->arg_dict()[key].SyncCopyFromCPU(feature_outputs[i]);
      exec_->arg_dict()[key].WaitToRead();
    }
    // Do forward prediction.
    exec_->Forward(false);

    vector<NDArray> outputs = exec_->outputs;
    // Convert output into score_matrix.
    ScoreMatrix scoreMatrix;
    size_t size = outputs[0].Size();
    scoreMatrix.data_ptr_ = scoreMatrixDptr;  // Change to Fixed Array.
    scoreMatrix.row_ = batch_size_ * beam_size_;
    scoreMatrix.col_ = num_actions_;
    outputs[0].SyncCopyToCPU(scoreMatrix.data_ptr_, size);
    // PopulateFeatureOutputs & Get All Beam States.
    feature_outputs.clear();
    beam_parser_->Compute(context, beam_reader_->batch_state_.get(),
                          scoreMatrix, feature_outputs, all_alive);
    Padding(feature_outputs);

    accumulate_steps += 1;
  }
  LOG(INFO) << "accumulate_steps: [" << accumulate_steps << "]";
  beam_eval_output_->Compute(context, beam_reader_->batch_state_.get(), documents);
  return epoch;
}
