#ifndef DOCUMENT_FORMAT_H_
#define DOCUMENT_FORMAT_H_

#include "sentence.h"

/*!
 * \brief A document format component converts a key/value pair from
 * a record to one or more documents.
 */
class DocumentFormat : public RegisterableClass<DocumentFormat> {
  public:
    DocumentFormat() {}
    virtual ~DocumentFormat() {}

    // Reads a record from the given input buffer with format specific logic.
    // Returns false if no record could be read because we reached end of file.
    virtual bool ReadRecord(tensorflow::io::InputBuffer *buffer,
        string *record) = 0;

    // Converts a key/value pair to one or more documents.
    virtual void ConvertFromString(const string &key, const string &value,
        vector<Sentence *> *documents) = 0;

    // Converts a document to a key/value pair.
    virtual void ConvertToString(const Sentence &document,
        string *key, string *value) = 0;
};

#endif
