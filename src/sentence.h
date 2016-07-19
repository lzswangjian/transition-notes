#ifndef SENTENCE_H
#define SENTENCE_H

#include "base.h"

class Token {
public:
  void set_word(const std::string &word) { word_ = word; }
  const string &word() const { return word_; }

  void set_start(int32_t start) { start_ = start; }
  int32_t start() const { return start_; }

  void set_end(int32_t end) { end_ =end; }
  int32_t end() const { return end_; }

  void set_head(int32_t head) { head_ = head; }
  int32_t head() const { return head_; }

  void set_tag(const std::string &tag) { tag_ = tag; }
  const string &tag() const { return tag_; }

  void set_category(const std::string &category) { category_ = category; }
  const string &category() const  { return category_; }

  void set_label(const std::string &label) { label_ = label; }
  const string &label() const { return label_; }

private:
  // Token word form.
  string word_;

  // Start & End position of token in text.
  int32_t start_;
  int32_t end_;

  // head index.
  int32_t head_;

  // Part-of-Speech tag for token.
  string tag_;

  // Coarse-grained word category for token.
  string category_;

  // Label for dependency relation between this token and its head.
  string label_;

  enum BreakLevel {
    NO_BREAK = 0,
    SPACE_BREAK = 1,
    LINE_BREAK = 2,
    SENTENCE_BREAK = 3,
  };

  BreakLevel break_level_;
};

class Sentence {
public:
  void set_docid(const std::string &docid) { docid_ = docid; }
  const std::string &docid() const { return docid_; }

  void set_text(const std::string &text) { text_ = text; }
  const std::string &text() const { return text_; }

  void set_token(const std::vector<Token *> &token) {
    token_ = token;
  }

  Token &token(int index) const { return *token_[index]; }

  int token_size() const { return token_.size(); }

  Token *add_token() {
    Token *token = new Token();
    token_.push_back(token);
    return token;
  }

public:
  Sentence() {}

  ~Sentence() {
    for (size_t i = 0; i != token_size(); ++i) {
      delete token_[i];
    }
    token_.clear();
  }


private:
  std::string docid_;
  std::string text_;
  std::vector<Token *> token_;
};


#endif /* end of include guard: SENTENCE_H */
