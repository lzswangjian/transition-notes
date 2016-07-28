/*!
 * \brief Tagger transition system.
 * This transition system has one type of actions:
 * - The SHIFT action pushes the next input token to the stack and
 *   advances to the next input token, assigning a part-of-speech tag
 *   to the token that was shifted.
 *
 * The transition system operates with parser actions encoded as integers:
 * - A SHIFT action is encoded as number starting from 0.
 */
#include <string>
#include <vector>

#include "parser_state.h"
#include "parser_transitions.h"
#include "../lexicon/term_frequency_map.h"
#include "../utils/shared_store.h"


class TaggerTransitionState : public ParserTransitionState {
  public:
    explicit TaggerTransitionState(const TermFrequencyMap *tag_map,
                                   const TagToCategoryMap *tag_to_category)
    : tag_map_(tag_map), tag_to_category_(tag_to_category) {}

    explicit TaggerTransitionState(const TaggerTransitionState *state)
    : TaggerTransitionState(state->tag_map_, state->tag_to_category_){
        tag_ = state->tag_;
        gold_tag_ = state->gold_tag_;
    }

    ParserTransitionState *Clone() const override {
      return new TaggerTransitionState(this);
    }

    void Init(ParserState *state) override {
      tag_.resize(state->sentence().token_size(), -1);
      gold_tag_.resize(state->sentence().token_size(), -1);
      for (int pos = 0; pos < state->sentence().token_size(); ++pos) {
        int tag = tag_map_->LookupIndex(state->GetToken(pos).tag(), -1);
        gold_tag_[pos] = tag;
      }
    }

    // Returns the tag assigned to a given token.
    int Tag(int index) const {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, tag_.size());
      return index == -1 ? -1 : tag_[index];
    }

    // Sets this tag on the token at index.
    void SetTag(int index, int tag) {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, tag_.size());
      tag_[index] = tag;
    }

    // Returns the gold tag for a given token.
    int GoldTag(int index) const {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, tag_.size());
      return index == -1 ? -1 : gold_tag_[index];
    }

    // Returns the string representation of a POS tag, or an empty string
    // if the tag is invalid.
    std::string TagAsString(int tag) const {
      if (tag >= 0 && tag < tag_map_->Size()) {
        return tag_map_->GetTerm(tag);
      } else {
        return "";
      }
    }

    // Adds transition state specific annotations to the document.
    void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
        Sentence *sentence) const override {
    }

    bool IsTokenCorrect(const ParserState &state, int index) const override {
      return GoldTag(index) == Tag(index);
    }

    // Returns a human readable string representation of this state.
    std::string ToString(const ParserState &state) const override {
      string str;
      for (int i = state.StackSize(); i > 0; --i) {
        const string &word = state.GetToken(state.Stack(i - 1)).word();
        if (i != state.StackSize() - 1) str.append(" ");
        str.append(word).append("[").append(TagAsString(Tag(state.StackSize() - i))).append("]");
      }
      for (int i = state.Next(); i < state.NumTokens(); ++i) {
        str.append(" ").append(state.GetToken(i).word());
      }
      return str;
    }

  private:
    // Currently assigned POS tags for each token in this sentence.
    std::vector<int> tag_;

    // Gold POS tags from the input document.
    std::vector<int> gold_tag_;

    // Tag map used for conversions between integer and string representations
    // part of speech tags.
    const TermFrequencyMap *tag_map_ = nullptr;

    // Tag to category map.
    const TagToCategoryMap *tag_to_category_ = nullptr;
};


class TaggerTransitionSystem : public ParserTransitionSystem {
  public:
    ~TaggerTransitionSystem() override {}

    void Setup(TaskContext *context) override {
        input_tag_map_ = context->GetInput("tag-map");
    }
    
    void Init(TaskContext *context) override {
        const string tag_map_path = TaskContext::InputFile(*input_tag_map_);
        tag_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
                tag_map_path, 0, 0);
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

  // Checks if the action is allowed in a given parser state.
    bool IsAllowedAction(ParserAction action,
                         const ParserState &state) const override {
      return !state.EndOfInput();
    }

  // Makes a shift by pushing the next input token on the stack and moving
  // to the next position.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override {
    if (!state->EndOfInput()) {
      MutableTransitionState(state)->SetTag(state->Next(), action);
      state->Push(state->Next());
      state->Advance();
    }
  }

  // We are in a final state when we reached the end of the input and the stack
  // is empty.
  bool IsFinalState(const ParserState &state) const override {
    return state.EndOfInput();
  }

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override {
    return "SHIFT(" + tag_map_->GetTerm(action) + ")";
  }

  // No state is deterministic in this transition system.
  bool IsDeterministicState(const ParserState &state) const override {
    return false;
  }

  // Returns a new transition state to be used to enhance the parser state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    return new TaggerTransitionState(tag_map_, tag_to_category_);
  }

  static const TaggerTransitionState &TransitionState(const ParserState &state) {
    return *static_cast<const TaggerTransitionState *>(state.transition_state());
  }

  // Downcasts the ParserTransitionState in ParserState to an TaggerTransitionState.
  static TaggerTransitionState *MutableTransitionState(ParserState *state) {
    return static_cast<TaggerTransitionState *>(state->mutable_transition_state());
  }

    // Input for the tag map. Not owned.
    TaskInput *input_tag_map_ = nullptr;

    // Tag map, Owned through SharedStore
    const TermFrequencyMap *tag_map_ = nullptr;

    // Tag to category map. Owned through SharedStore.
    const TagToCategoryMap *tag_to_category_ = nullptr;
};

