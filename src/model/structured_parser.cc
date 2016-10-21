#include "structured_parser.h"

StructuredParser::StructuredParser(int batch_size)
  : GreedyParser(batch_size) {
  // Do Initialization
  max_steps_ = 25;
}

StructuredParser::~StructuredParser() {

}

void StructuredParser::TrainIter() {
  int accumulate_steps = 0;
  bool all_alive = true;

  while (accumulate_steps < max_steps_ & all_alive) {
    // Predict
    exec_->Forward(true);

    // PopulateFeatureOutputs.
    beam_parser_->Compute(nullptr);

    // Do Predict (use Greedy Parser interface?)

    // Get All Beam States.
  }

  // Make Cross Entropy Loss.

  // Update Parameters.
}

Symbol StructuredParser::BuildSequence() {
  return Symbol();
}
