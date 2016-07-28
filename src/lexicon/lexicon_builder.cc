#include <string>

#include "../utils/utils.h"
#include "term_frequency_map.h"
#include "../sentence.h"
#include "../options.h"
#include "../io/text_formats.h"

/*!
 * \brief A workflow task that creates term maps. (e.g., word, tag, etc.).
 */
class LeiconBuilder {
public:
    void Compute(Options &options) {
        // Term frequency maps to be populated by the corpus.
        TermFrequencyMap words;
        TermFrequencyMap lcwords;
        TermFrequencyMap tags;
        TermFrequencyMap categories;
        TermFrequencyMap labels;

        int64_t num_tokens = 0;
        int64_t num_documents = 0;

        // Read
        ifstream m_file(options.input_file_);
        if (!m_file.is_open()) {
            LOG(INFO) << "Open file [" << options.input_file_ << "] failed.";
            return;
        }
        CoNLLSyntaxFormat readParser;
        string record;
        string doc_id = "conll";
        vector<Sentence *> sentences;
        while (readParser.ReadRecord(&m_file, &record)) {
            readParser.ConvertFromString(doc_id, record, &sentences);
        }
        m_file.close();

        for (size_t i = 0; i < sentences.size(); ++i) {
            Sentence *document = sentences[i];
            for (int t = 0; t < document->token_size(); ++t) {
                Token &token = document->token(t);
                string word = token.word();
                utils::NormalizeDigits(&word);
                string lcword = utils::Lowercase(word);

                CHECK(lcword.find('\n') == string::npos);
                if (!word.empty() && !HasSpaces(word)) words.Increment(word);
                if (!lcword.empty() && !HasSpaces(lcword)) lcwords.Increment(lcword);
                if (!token.tag().empty()) tags.Increment(token.tag());
                if (!token.category().empty()) categories.Increment(token.category());
                if (!token.label().empty()) labels.Increment(token.label());

                ++num_tokens;
            }
            ++num_documents;
        }

        LOG(INFO) << "Term maps collected over " << num_tokens << " tokens from "
                  << num_documents << " documents.";

        // Save into file.
        words.Save(options.word_map_file_);
        lcwords.Save(options.lc_word_map_file_);
        tags.Save(options.tag_map_file_);
        categories.Save(options.category_map_file_);
        labels.Save(options.label_map_file_);
    }

private:
    // Returns true if the word contains spaces.
    static bool HasSpaces(const string &word) {
        for (char c : word) {
            if (c == ' ') return true;
        }
        return false;
    }
};


class FeatureSize {
public:
    explicit FeatureSize() {
    }
};
