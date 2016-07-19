/*!
 * \brief Arc-standard transition system.
 *
 * This transition system has three types of actions:
 *  SHIFT: It pushes the next input token to the stack and advances to the next input token.
 *  LEFT_ARC: It adds a dependency relation from first to second token on the stack and removes second one.
 *  RIGHT_ARC: It adds a dependency relation from second to first token on the stack and removes first one.
 *
 *  The transition system operates with parser actions encoded as intergers:
 *  SHIFT: encoded as 0.
 *  LEFT_ARC: encoded as an odd number starting from 1.
 *  RIGHT_ARC: encoded as an even number starting from 2.
 */

#include <string>

#include "parser_state.h"
#include "parser_transitions.h"

using namespace std;

class ArcStandardTransitionState : public ParserTransitionState {
public:
  ParserTransitionState *Clone() const override {
    return new ArcStandardTransitionState();
  }

  // Pushes the root on the stack before using the parser state in parsing.
  void Init(ParserState *state) override { state->Push(-1); }

  // Adds transition state specific annotations to the document.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    for (int i = 0; i < state.NumTokens(); ++i) {
      Token *token = sentence->mutable_token(i);
      token->set_label(state.LabelAsString(state.Label(i)));
      if (state.Head(i) != -1) {
        token->set_head(state.Head(i));
      } else {
        token->clear_head();
        if (rewrite_root_labels) {
          token->set_label(state.LabelAsString(state.RootLabel()));
        }
      }
    }
  }

  // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return state.GoldHead(index) == state.Head(index);
  }

  // Returns a human readable string representation of this state.
  string ToString(const ParserState &state) const override {
  }
};

class ArcStandardTransitionSystem : public ParserTransitionSystem {
public:
  // Action types for the arc-standard transition system.
  enum ParserActionType {
    SHIFT = 0,
    LEFT_ARC = 1,
    RIGHT_ARC = 2,
  };

  static ParserAction ShiftAction() { return SHIFT; }

  static ParserAction LeftArcAction(int label) { return 1 + (label << 1); }

  static ParserAction RightArcAction(int label) { return 1 + ((label << 1) | 1); }

  static ParserActionType ActionType(ParserAction action) {
  }

  int NumActionTypes() const override { return 3; }

  int NumActions(int num_labels) const override {
    return 1 + 2 * num_labels;
  }

  // Returns the default action for a given state.
  ParserAction GetDefaultAction(const ParserState &state) const override {
    // If there are further tokens available in the input then Shift.
    if (!state.EndOfInput()) {
      return ShiftAction();
    } else {
      // Do a "reduce".
      return RightArcAction(2);
    }
  }

  // Returns the next gold action for a given state according to the underlying
  // annotated sentence.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    // If the stack contains less than 2 tokens, the only valid parser action is
    // shift.
    if (state.StackSize() < 2) {
      DCHECK(!state.EndOfInput());
      return ShiftAction();
    }

    // If the second token on the stack is the head of the first one, return a right
    // arc action.
    if (state.GoldHead(state.Stack(0)) == state.Stack(1) &&
        DoneChildrenRightOf(state, state.Stack(0))) {
      const int gold_label = state.GoldLabel(state.Stack(0));
      return RightArcAction(gold_label);
    }

    // If the first token on the stack is the head of the second one, return a left arc
    // action.
    if (state.GoldHead(state.Stack(1)) == state.Top()) {
      const int gold_label = state.GoldLabel(state.Stack(1));
      return LeftArcAction(gold_label);
    }

    // Otherwise, shift.
    return ShiftAction();
  }

  // Determines if a token has any children to the right in the sentence.
  // Arc standard is a bottom-up parsing method and has to finish all sub-trees
  // first.
  static bool DoneChildrenRightOf(const ParserState &state, int head) {
  }


};
