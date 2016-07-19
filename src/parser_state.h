#ifndef PARSER_STATE_H_
#define PARSER_STATE_H_

#include <string>
#include <vector>

#include "parser_transitions.h"
#include "sentence.h"

class TermFrequencyMap;

/*!
 * \brief A ParserState object represents the state of the parser during the
 * parsing of a sentence. The state consists of a pointer to the next input
 * token and a stack of partially processed tokens.
 */
class ParserState {
  public:
    static const char kRootLabel[];

    static const int kDefaultRootLabel = -1;

    ParserState(Sentence *sentence,
        ParserTransitionState *transition_state,
        const TermFrequencyMap *label_map);

    ~ParserState();

    // Clones the parser state.
    ParserState *Clone() const;

    // Clones the parser state.
    int RootLabel() const;

    // Returns the index of the next input token.
    int Next() const;

    // Returns the number of tokens in the sentence.
    int NumTokens() const { return num_tokens_; }

    // Returns
    int Input(int offset) const;

    // Adcances to the next input token.
    void Advance();

    // Returns true if all tokens have been processed.
    bool EndOfInput() const;

    // Pushes an element to the stack.
    void Push(int index);

    // Pops the top element from stack and returns it;
    int Pop();

    // Returns the element from the top of the stack.
    int Top() const;

    // Returns the element at a certain position in the stack. Stack(0) is 
    // the top stack element. If no such position exists, returns -2.
    int Stack(int position) const;

    int StackSize() const;

    bool StackEmpty() const;

    // Returns the head index for a given token.
    int Head(int index) const;

    // Returns the label of the relation to head for a given token.
    int Label(int index) const;

    // Returns the parent of a given token 'n' levels up in the tree.
    int Parent(int index, int n) const;

    int LeftmostChild(int index, int n) const;

    int RightmostChild(int index, int n) const;

    int LeftSibling(int index, int n) const;

    int RightSibling(int index, int n) const;

    void AddArc(int index, int head, int label);

    bool IsTokenCorrect(int index) const;

    int GoldHead(int index) const;

    int GoldLabel(int index) const;

    const Token &GetToken(int index) const {
      if (index == -1) {
        return kRootToken;
      } else {
        return sentence().token(index);
      }
    }

    // Annotates a document with the dependency relations build during parsing
    // for one of its sentences. If rewrite_root_labels is true, then all
    // tokens with no heads will be assigned the default root label "ROOT".
    void AddParseToDocument(Sentence *document, bool rewrite_root_labels) const;
    void AddParseToDocument(Sentence *document) const {
      AddParseToDocument(document, true);
    }

    // Returns the string representation of a dependency label, or an empty 
    // string if the label is invalid.
    std::string LabelAsString(int label) const;

    // Returns a string representation of the parser state.
    std::string ToString() const;

    // Returns the underlying sentence instance.
    const Sentence &sentence() const { return *sentence_; }
    Sentence *mutable_sentence() const { return sentence_; }

    // Returns the transiton system-specific state.
    const ParserTransitionState *transition_state() const {
      return transition_state_;
    }
    ParserTransitionState *mutable_transition_state() const {
      return transition_state_;
    }

    // Gets/sets the flag which says that the state was obtained through
    // gold transitions only.
    bool is_gold() const { return is_gold_; }
    void set_is_gold(bool is_gold) { is_gold_ = is_gold; }

  private:
    ParserState() {}

    // Default value for the root token.
    Token kRootToken;

    // Sentence to parse.
    Sentence *sentence_ = nullptr;

    // Number of tokens in the sentence to parse.
    int num_tokens_;

    int alternative_ = -1;

    // Transition system-specific state.
    ParserTransitionState *transition_state_ = nullptr;

    // Label map used for conversions between integer and string representations
    // of the dependency labels.
    const TermFrequencyMap *label_map_ = nullptr;

    // Root label.
    int root_label_;

    // Index of the next input token.
    int next_;

    // Parse stack of partially processed tokens.
    std::vector<int> stack_;

    // List of head positions for the (partial) dependency tree.
    std::vector<int> head_;

    // List of dependency relation labels describing the (partial) dependency.
    std::vector<int> label_;

    // Score for the parser state.
    double score_ = 0.0;

    // True if this is the gold standard sequence (used for structured learning).
    bool is_gold_ = false;
};

#endif
