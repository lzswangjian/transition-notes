#include "sentence_batch.h"
#include "io/text_reader.h"

void SentenceBatch::Init(TaskContext *context) {
    reader_.reset(new TextReader(*context->GetInput(input_name_)));
    size_ = 0;
}

bool SentenceBatch::AdvanceSentence(int index) {
    if (sentences_[index] == nullptr) ++size_;

    Sentence* sentence = reader_->Read();
    if (sentence == nullptr) {
        --size_;
        return false;
    }

    // Preprocess the new sentence for the parser state.
    sentences_[index] = sentence;
    return true;
}
