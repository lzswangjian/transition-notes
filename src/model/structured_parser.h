#ifndef MODEL_STRUCTURED_PARSER_H_
#define MODEL_STRUCTURED_PARSER_H_

#include "greedy_parser.h"
#include "../beam_reader_ops.h"

/*!
 * \brief Extends GreedyParser with beam search.
 * \author Sheng Li
 */
class StructuredParser : public GreedyParser {
 public:
  StructuredParser(int batch_size);

  StructuredParser(int batch_size,
                   int num_actions,
                   vector<int> &num_features,
                   vector<int> &num_feature_ids,
                   vector<int> &embedding_sizes,
                   vector<int> &hidden_layer_sizes);

  virtual ~StructuredParser();

 public:
  void BuildSequence();
  void CrossEntropy(vector<NDArray> &step_head_ndarray, int accumulate_steps);
  void SetupModel(Symbol symbol);

  // Init Parameters from Pretrained Model.
  void InitWithPreTrainedParameters(const string &param_path);
  // Init Parameters from scratch.
  void InitFreshParameters();

  void Padding(vector<vector<float>> &feature_outputs);

  inline void Softmax(vector<float> &energy, vector<float> &out) {
    out.clear();
    out.resize(energy.size());
    float mmax = energy[0];
    for (size_t i = 1; i < energy.size(); ++i) {
      mmax = std::max(mmax, energy[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < energy.size(); ++i) {
      out[i] = std::exp(energy[i] - mmax);
      sum += out[i];
    }

    for (size_t i = 0; i < energy.size(); ++i) {
      out[i] /= sum;
    }
  }

  inline void SoftmaxGrad(vector<float> &src, vector<float> &grad, int label) {
    grad.clear();
    grad.resize(src.size());

    for (int i = 0; i < src.size(); ++i) {
      if (i == label) {
        grad[i] = src[i] - 1.0f;
      } else {
        grad[i] = src[i];
      }
    }
  }

  void CreateOptimizer(const std::string &opt = "ccsgd");

 public:
  int TrainIter();

 public:
  BeamParseReader *beam_reader_ = nullptr;
  BeamParser *beam_parser_ = nullptr;
  BeamParserOutput *beam_parser_output_ = nullptr;
  int max_steps_;
  int beam_size_;
  std::vector<Executor *> exec_list_;
  TaskContext *context = nullptr; // Not Owned.
  float *scoreMatrixDptr = nullptr; // Owned.
};

#endif
