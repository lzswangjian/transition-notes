#include "sentence_features.h"

FeatureValue Hyphen::ComputeValue(const Token &token) const {
  const string &word = token.word();
  // bug? if no '-', find function will return -1
  return (word.find('-') < word.length() ? HAS_HYPHEN : NO_HYPHEN);
}
