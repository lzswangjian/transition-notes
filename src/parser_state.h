#ifndef PARSER_STATE_H_
#define PARSER_STATE_H_

class ParserState {
  public:
    static const char kRootLabel[];

    static const int kDefaultRootLabel = -1;

    ParserState(Sentence *sentence,
        ParserTransitionState *transition_state,
        const TermFrequencyMap *label_map);

    ~ParserState();

    ParserState *Clone() const;

  private:
    ParserState() {}

    // Default value for the root token.
    const Token kRootToken;

    // Sentence to parse.
    Sentence *sentence_ = nullptr;

    // Number of tokens in the sentence to parse.
    int num_tokens_;

    int alternative_ = -1;

    // Transition system-specific state.
    ParserTransitionState *transition_state_ = nullptr;

    // Root label.
    int root_label_;

    // Index of the next input token.
    int next_;

    // Parse stack of partially processed tokens.
    vector<int> statck_;

    // List of head positions for the (partial) dependency tree.
    vector<int> head_;

    // List of dependency relation labels describing the (partial) dependency.
    vector<int> label_;

    // Score for the parser state.
    double score_ = 0.0;

    // True if this is the gold standard sequence (used for structured learning).
    bool is_gold_ = false;
};

#endif
