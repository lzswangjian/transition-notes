#include <memory>
#include <string>
#include <vector>

/*!
 * \brief CoNLL document format reader for dependency annotated corpora.
 *
 * Fileds:
 * 1 ID: Token counter, starting at 1 for each new sentence and increasing
 *       by 1 for every new token.
 * 2 FORM: Word form or punctuation symbol.
 * 3 LEMMA: Lemma or stem.
 * 4 CPOSTAG: Coarse-grained part-of-speech tag or category.
 * 5 POSTAG: Fine-grained part-of-speech tag.
 * 6 FEATS: Unordered set of syntactic and/or morphological features.
 * 7 HEAD: Head of the current token, which is either a value of ID or '0'.
 * 8 DEPREL: Dependency relation to the HEAD.
 * 9 PHEAD: Projective head of current token.
 * 10 PDEPREL: Dependency relation to the PHEAD.
 */

class CoNLLSyntaxFormat : public DocumentFormat {
  public:
    CoNLLSyntaxFormat() {}

    // Reads up to the first empty line and returns false end of file is reached.
    bool ReadRecord(tensorflow::io::InputBuffer *buffer,
        string *record) override {
    }

    void ConvertFromString(const string &key, const string &value,
        vector<Sentence *> *sentences) override {
    }

    void ConvertToString(const Sentence &sentence, string *key,
        string *value) override {
    }
};
