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
  virtual ~StructuredParser();

 public:
  Symbol BuildSequence();

 public:
  void TrainIter();

 public:
  BeamParser *beam_parser_ = nullptr;
  int max_steps_;
};

#endif