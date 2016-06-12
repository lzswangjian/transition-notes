#ifndef PARSER_TRANSITIONS_H_
#define PARSER_TRANSITIONS_H_

typedef int ParserAction;

enum class LabelType {
  NO_LABEL = 0,
  LEFT_LABEL = 1,
  RIGHT_LABEL = 2,
};

/*!
 * \brief Transition system-specfic state.
 * Transition systems can subclass this to preprocess the parser
 * state and/or to keep additional information during parsing.
 */
class ParserTransitionState {
  public:
    virtual ~ParserTransitionState() {}

    // Clones the transition state.
    virtual ParserTransitionState *Clone() const = 0;

    // Initializes a parser state for the transition system.
    virtual void Init(ParserState *state) = 0;

    virtual void AddParseToDocument(const ParserState &state,
        bool rewrite_root_labels,
        Sentence *sentence) const {}

    virtual string ToString(const ParserState &state) const = 0;
};


class ParserTransitionSystem {
  public:
    ParserTransitionSystem() {}
    virtual ~ParserTransitionSystem() {}

    virtual void Setup() {}
    
    virtual void Init() {}

    virtual void Read() {}

    virtual void Write() {}

    // Return the number of actions types.
    virtual int NumActionTypes() const = 0;

    // Return the number of actions.
    virtual int NumActions(int num_labels) const = 0;

    // Internally creates the set of outcomes (when transition systems
    // support a variable number of actions).
    virtual void CreateOutcomSet(int num_labels) {}

    // Returns the default action for a given state.
    virtual ParserAction GetDefaultAction(const ParserState &state) const = 0;

    // Returns the next gold action for the parser during training using the
    // dependency relations found in the underlying annotated sentence.
    virtual ParserAction GetNextGoldAction(const ParserState &state) const = 0;

    // Returns all next gold actions for the parser during training using the
    // dependency relations found in the underlying annotated sentence.
    virtual void GetAllNextGoldActions(const ParserState &state,
        vector<ParserState> *actions) const {
      ParserAction action = GetNextGoldAction(state);
      *actions = {action};
    }

    // Returns the number of atomic actions within the specified ParserAction.
    virtual int ActionLength(ParserAction action) const { return 1; }

    // Returns true if the action is allowed in the given parser state.
    virtual bool IsAllowedAction(ParserAction action,
        const ParserState &state) const = 0;

    // Performs the specified action on a given parser state. The action is
    // saved in the state's history.
    void PerformAction(ParserAction action, ParserState *state) const;

    // Performs the specified action on a given parser state. The action is not
    // saved in the state's history.
    virtual void PerformActionWithoutHistory(ParserAction action,
        ParserState *state) const;

    // Returns true if a given state is deterministic.
    virtual bool IsDeterministicState(const ParserState &state) const;

    // Returns true if no more actions can be applied to a given parser state.
    virtual bool IsFinalState(const ParserState &state) const = 0;

    // Returns a string representation of a parser action.
    virtual string ActionAsString(ParserAction action,
        const ParserState &state) const = 0;

    // Get the index of the child that would be created by this action.
    // -1 for no child created.
    virtual int ChildIndex(const ParserState &state,
        const ParserAction &action) const {
      return -1;
    }

    // Get the index of the parent that would gain a new child by this action.
    // -1 for no parent modified.
    virtual int ParentIndex(const ParserState &state,
        const ParserAction &action) const {
      return -1;
    }
};

#endif
