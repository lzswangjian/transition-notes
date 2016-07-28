#include "parser_features.h"

#include "sentence_features.h"

// Registry for the parser feature functions.
REGISTER_CLASS_REGISTRY("parser feature function", ParserFeatureFunction);

// Registry for the parser state + token index feature functions.
REGISTER_CLASS_REGISTRY("parser+index feature function", ParserIndexFeatureFunction);

RootFeatureType::RootFeatureType(const string &name, const FeatureType &wrapped_type,
                                 int root_value)
    : FeatureType(name), wrapped_type_(wrapped_type), root_value_(root_value) {}

string RootFeatureType::GetFeatureValueName(FeatureValue value) const {
    if (value == root_value_) return "<ROOT>";
    return wrapped_type_.GetFeatureValueName(value);
}

FeatureValue RootFeatureType::GetDomainSize() const {
    return wrapped_type_.GetDomainSize() + 1;
}
/*!
 * \brief Parser feature locator for accessing the remaining input tokens in the parser state.
 * It takes the offset relative to the current input token as argument. Negative values represent
 * tokens to the left, positive values to the right and 0 (the default argument value) represents
 * the current input token.
 */
class InputParserLocator : public ParserLocator<InputParserLocator> {
public:
  // Get the new focus.
  int GetFocus(const WorkspaceSet &workspaces, const ParserState &state) const {
    const int offset = argument();
    return state.Input(offset);
  }
};

REGISTER_PARSER_FEATURE_FUNCTION("input", InputParserLocator);

/*!
 * \brief Parser feature locator for accessing the stack in the parser state.
 * The argument represents the position on the stack, 0 being the top of the stack.
 */
class StackParserLocator : public ParserLocator<StackParserLocator> {
public:
  int GetFocus(const WorkspaceSet &workspaces, const ParserState &state) const {
    const int position = argument();
    return state.Stack(position);
  }
};

REGISTER_PARSER_FEATURE_FUNCTION("stack", StackParserLocator);

/*!
 * \brief Parser feature locator for locating the head of the focus token. The
 * argument specifies the number of times the head function is applied. Please
 * note that this operates on partially built dependency trees.
 */
class HeadFeatureLocator : public ParserIndexLocator<HeadFeatureLocator> {
public:
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
    if (*focus < -1 || *focus >= state.sentence().token_size()) {
      *focus = -2;
      return;
    } else {
      const int levels = argument();
      *focus = state.Parent(*focus, levels);
    }
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("head", HeadFeatureLocator);

/*!
 * \brief Parser feature locator for locating children of the focus token. The
 * argument specifies the number of times the leftmost (when the argument is < 0)
 * or rightmost (when the argument > 0) child function is applied. Please note
 * that this operates on partially built dependency trees.
 */
class ChildFeatureLocator : public ParserIndexLocator<ChildFeatureLocator> {
public:
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
    if (*focus < -1 || *focus >= state.sentence().token_size()) {
        *focus = -2;
        return;
    } else {
        const int levels = argument();
        if (levels < 0) {
            *focus = state.LeftmostChild(*focus, -levels);
        } else {
            *focus = state.RightmostChild(*focus, levels);
        }
    }
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("child", ChildFeatureLocator);

/*!
 * \brief Parser feature locator for locating siblings of the focus token.
 */
class SiblingFeatureLocator
  : public ParserIndexLocator<SiblingFeatureLocator> {
public:
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
    if (*focus < -1 || *focus >= state.sentence().token_size()) {
        *focus = -2;
        return;
    } else {
        const int position = argument();
        if (position < 0) {
            *focus = state.LeftSibling(*focus, -position);
        } else {
            *focus = state.RightSibling(*focus, position);
        }
    }
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("sibling", SiblingFeatureLocator);

// Feature function for computing the label from focus token. Note that this
// does not use the precomputed values, since we get the labels from the stack;
// the reason it utilizes sentence_features::Label is to obtain the label map.
class LabelFeatureFunction
  : public BasicParserSentenceFeatureFunction<Label> {
public:
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
      if (focus == -1) return RootValue();
      if (focus < -1 || focus >= state.sentence().token_size()) {
          return feature_.NumValues();
      }
      const int label = state.Label(focus);
      return label == -1 ? RootValue() : label;
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("label", LabelFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Word> WordFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("word", WordFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Tag> TagFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("tag", TagFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Digit> DigitFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("digit", DigitFeatureFunction);


// Parser feature function that can use nested sentence feature functions for
// feature extraction.
class ParserTokenFeatureFunction : public NestedFeatureFunction<
  FeatureFunction<Sentence, int>, ParserState, int> {
public:
  void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
      for (auto *function : nested_) {
          function->Preprocess(workspaces, state->mutable_sentence());
      }
  }

  void Evaluate(const WorkspaceSet &workspaces, const ParserState &state,
                int focus, FeatureVector *result) const override {
      for (auto *function : nested_) {
          function->Evaluate(workspaces, state.sentence(), focus, result);
      }
  }

  // Returns the first nested feature's computed value.
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
      if (nested_.empty()) return -1;
      return nested_[0]->Compute(workspaces, state.sentence(), focus, result);
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("token", ParserTokenFeatureFunction);

