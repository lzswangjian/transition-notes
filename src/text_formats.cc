#include <memory>
#include <string>
#include <vector>

#include "sentence.h"
#include "utils.h"

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
class CoNLLSyntaxFormat {
  public:
    CoNLLSyntaxFormat() {}

    // Reads up to the first empty line and returns false end of file is reached.
    bool ReadRecord(ifstream &stream,
        string *record) {
      string line;
      record->clear();
      while (std::getline(stream, line) && !line.empty()) {
        record->append(line);
        record->append("\n");
      }
      return !record->empty();
    }

    void ConvertFromString(const string &key, const string &value,
        vector<Sentence *> *sentences) {
      // Create new sentence.
      Sentence *sentence = new Sentence();

      // Each line corresponds to one token.
      string text;
      vector<string> lines = utils::Split(value, '\n');

      // Add each token to the sentence.
      vector<string> fields;
      int expected_id = 1;
      for (size_t i = 0; i < lines.size(); ++i) {
        // Split line into tab-separated fields.
        fields.clear();
        fields = utils::Split(lines[i], '\t');
        if (fields.size() == 0) continue;

        // Skip comment lines.
        if (fields[0][0] == '#') continue;

        // Check that the line is valid.
        CHECK_GE(fields.size(), 8)
          << "Every line has to have at least 8 tab separated fields.";

        // Check that the ids follow the expected format.
        const int id = utils::ParseUsing<int>(fields[0], 0, utils::ParseInt32);
        CHECK_EQ(expected_id++, id)
          << "Token ids start at 1 for each new sentence and increase by 1 "
          << "on each new token. Sentences are separated by an empty line.";

        // Get relevant fields.
        const string &word = fields[1];
        const string &cpostag = fields[3];
        const string &tag = fields[4];
        const int head = utils::ParseUsing<int>(fields[6], 0, utils::ParseInt32);
        const string &label = fields[7];

        // Add token to sentence text.
        if (!text.empty()) text.append(" ");
        const int start = text.size();
        const int end = start + word.size() - 1;
        text.append(word);

        // Add token to sentence.
        Token *token = sentence->add_token();
        token->set_word(word);
        token->set_start(start);
        token->set_end(end);
        if (head > 0) token->set_head(head - 1);
        if (!tag.empty()) token->set_tag(tag);
        if (!cpostag.empty()) token->set_category(cpostag);
        if (!label.empty()) token->set_label(label);
      }

      if (sentence->token_size() > 0) {
        sentence->set_docid(key);
        sentence->set_text(text);
        sentences->push_back(sentence);
      } else {
        // If the sentence was empty (e.g., blank lines at the begining of a 
        // file), then don't save it.
        delete sentence;
      }
    }

    // Converts a sentence to a key/value pair.
    void ConvertToString(const Sentence &sentence, string *key,
        string *value) {
      *key = sentence.docid();
      vector<string> lines;
      for (int i = 0; i < sentence.token_size(); ++i) {
        vector<string> fields(10);
        fields[0] = utils::Printf(i + 1);
        fields[1] = sentence.token(i).word();
        fields[2] = "_";
        fields[3] = sentence.token(i).category();
        fields[4] = sentence.token(i).tag();
        fields[5] = "_";
        fields[6] = utils::Printf(sentence.token(i).head() + 1);
        fields[7] = sentence.token(i).label();
        fields[8] = "_";
        fields[9] = "_";
        lines.push_back(utils::Join(fields, "\t"));
      }
      *value = utils::Join(lines, "\n") + "\n\n";
    }
};
