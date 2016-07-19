#include <string>
#include "parser_features.h"
#include "sentence_features.h"

// Registry for the parser feature functions.
REGISTER_CLASS_REGISTRY("parser feature function", ParserFeatureFunction);

// Registry for the parser state + token index feature functions.
REGISTER_CLASS_REGISTRY("parser+index feature function", ParserIndexFeatureFunction);

/*
 *
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


class ChildFeatureLocator : public ParserIndexLocator<ChildFeatureLocator> {
public:
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
  }
};

class SiblingFeatureLocator
  : public ParserIndexLocator<SiblingFeatureLocator> {
public:
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
  }
};

// Feature function for computing the label from focus token. Note that this
// does not use the precomputed values, since we get the labels from the stack;
// the reason it utilizes sentence_features::Label is to obtain the label map.
class LabelFeatureFunction
  : public BasicParserSentenceFeatureFunction<Label> {
public:
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("label", LabelFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Word> WordFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("word", WordFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Tag> TagFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("tag", TagFeatureFunction);

typedef BasicParserSentenceFeatureFunction<Digit> DigitFeatureFunction;
REGISTER_PARSER_IDX_FEATURE_FUNCTION("digit", DigitFeatureFunction);


class ParserTokenFeatureFunction : public NestedFeatureFunction<
  FeatureFunction<Sentence, int>, ParserState, int> {
public:
  void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
  }

  void Evaluate(const WorkspaceSet &workspaces, const ParserState &state,
                int focus, FeatureVector *result) const override {
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("token", ParserTokenFeatureFunction);

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




