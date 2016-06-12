/*!
 *\brief Tagger transition system.
 */
#include <string>
#include "parser_state.h"
#include "parser_transitions.h"


class TaggerTransitionState : public ParserTransitionState {
  public:
    void Init(ParserState *state) {
    }

    // Returns the tag assigned to a given token.
    int Tag(int index) const {
    }

    // Sets this tag on the token at index.
    void SetTag(int index, int tag) {
    }

    // Returns the string representation of a POS tag, or an empty string
    // if the tag is invalid.
    string TagAsString(int tag) const {
    }

    // Adds transition state specific annotations to the document.

  private:
    // Currently assigned POS tags for each token in this sentence.
    vector<int> tag_;

    // Gold POS tags from the input document.
    vector<int> gold_tag_;

    // Tag map used for conversions between integer and string representations
    // part of speech tags.
    const TermFrequencyMap *tag_map_ = nullptr;

    // Tag to category map.
    const TagToCategoryMap *tag_to_category_ = nullptr;
};


class TaggerTransitionSystem : public ParserTransitionSystem {
  public:
    ~TaggerTransitionSystem() {
    }

    void Init(TaskContext *context) {
    }

    // The SHIFT action uses the same value as the corresponding action type.
    static ParserAction ShiftAction(int tag) { return tag; }

    // Returns the number of action types.
    int NumActionTypes() const override { return 1; }

    // Returns the number of possible actions.
    int NumActions(int num_labels) const override { return tag_map_->Size(); }

    // The default action for a given state is assigning the most frequent tag.
    ParserAction GetDefaultAction(const ParserState &state) const override {
      return ShiftAction(0);
    }

    // Returns the next gold action for a given state according to the
    // underlying annotated sentence.
    ParserAction GetNextGoldAction(const ParserState &state) const override {
      if (!state.EndOfInput()) {
        return ShiftAction(TransitionState(state).GoldTag(state.Next()));
      } else {
        return ShiftAction(0);
      }
    }

    static const TaggerTransitionState &TransitionState(const ParserState &state) {
      return static_cast<TaggerTransitionState *>(state.transition_state());
    }

    // Tag map
    const TermFrequencyMap *tag_map_ = nullptr;

    // Tag to category map.
    const TagToCategoryMap *tag_to_category_ = nullptr;
};

