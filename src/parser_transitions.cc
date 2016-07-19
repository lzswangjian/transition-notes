#include "parser_transitions.h"

#include "parser_state.h"

void ParserTransitionSystem::PerformAction(ParserAction action,
    ParserState *state) const {
  PerformActionWithoutHistory(action, state);
}
