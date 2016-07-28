#include "text_reader.h"

TextReader::TextReader(const TaskInput &input) {
    file_name_ = TaskContext::InputFile(input);
    // format_.reset(DocumentFormat::Create(input.record_format()));
    format_.reset(new CoNLLSyntaxFormat());
    Reset();
}

Sentence *TextReader::Read() {
    vector<Sentence *> sentences;
    string key, value;
    while (sentences.empty() && format_->ReadRecord(file_, &value)) {
        key = file_name_ + ":" + utils::Printf(sentence_count_);
        format_->ConvertFromString(key, value, &sentences);
        CHECK_LE(sentences.size(), 1);
    }

    if (sentences.empty()) {
        return nullptr;
    } else {
        ++sentence_count_;
        return sentences[0];
    }

}

void TextReader::Reset() {
    if (file_ != nullptr) {
        file_->close();
        delete file_;
    }
    sentence_count_ = 0;
    file_ = new ifstream(file_name_);
    if (!file_->is_open()) {
        LOG(FATAL) << "Open file " << file_name_ << " failed.";
    }
}

TextReader::~TextReader() {
    if (file_ != nullptr) {
        file_->close();
        delete file_;
    }
}
