//
// Created by ShengLi on 7/23/16.
//

#ifndef SYNTAXNET_TEXT_READER_H
#define SYNTAXNET_TEXT_READER_H


#include "../sentence.h"
#include "document_format.h"
#include "text_formats.h"
#include "../utils/task_context.h"

class TextReader {
public:
    explicit TextReader(const TaskInput &input);
    ~TextReader();

    Sentence *Read();

    void Reset();

private:
    string file_name_;
    int sentence_count_ = 0;
    ifstream *file_ = nullptr;
    std::unique_ptr<DocumentFormat> format_;
};


#endif //SYNTAXNET_TEXT_READER_H
