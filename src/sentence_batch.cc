#include "sentence_batch.h"
#include "io/text_reader.h"

void SentenceBatch::Init(TaskContext *context) {
    reader_.reset(new TextReader(*context->GetInput(input_name_)));
    size_ = 0;
}

bool SentenceBatch::AdvanceSentence(int index) {
    if (sentences_[index] == nullptr) ++size_;
    sentences_[index].reset();
    std::unique_ptr<Sentence> sentence(reader_->Read());
    if (sentence == nullptr) {
        --size_;
        return false;
    }

    // Preprocess the new sentence for the parser state.
    sentences_[index] = std::move(sentence);
    return true;
}
