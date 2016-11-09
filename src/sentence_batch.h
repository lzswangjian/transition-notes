#ifndef SYNTAXNET_SENTENCE_BATCH_H_
#define SYNTAXNET_SENTENCE_BATCH_H_

#include "sentence.h"
#include "io/text_reader.h"
#include "utils/task_context.h"

/*!
 * \brief Helper class to manage generating batches of preprocessed ParserState objects
 * by reading in multiple sentences in parallel.
 */
class SentenceBatch {
 public:
  SentenceBatch(int batch_size, const string &input_name)
    : batch_size_(batch_size),
      input_name_(input_name),
      sentences_(batch_size) {}

  // Initializes all resources and opens the corpus file.
  void Init(TaskContext *context);

  // Advances the index'th sentence in the batch to the next sentence. This will
  // create and preprocess a new ParserState for that element. Returns false if
  // EOF is reached (if EOF, also sets the state to be nullptr.)
  bool AdvanceSentence(int index);

  // Rewinds the corpus reader.
  void Rewind() { reader_->Reset(); }

  int size() const { return size_; }

  Sentence *sentence(int index) { return sentences_[index].get(); }

 private:
  // Running tally of non-nullptr states in the batch.
  int size_;

  // Maximum number of states in the batch.
  int batch_size_;

  // Input to read from the TaskContext.
  string input_name_;

  // Reader for the corpus.
  std::unique_ptr<TextReader> reader_;

  // Batch: Sentence objects.
  std::vector<std::unique_ptr<Sentence>> sentences_;
};

#endif