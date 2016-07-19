#include "sentence_features.h"

TermFrequencyMapFeature::~TermFrequencyMapFeature() {

}

void TermFrequencyMapFeature::Setup(TaskContext *context) {
  TokenLookupFeature::Setup(context);
  context->GetInput(input_name_, "text", "");
}

void TermFrequencyMapFeature::Init(TaskContext *context) {
}

string TermFrequencyMapFeature::GetFeatureValueName(FeatureValue value) const {
  if (value == UnknownValue()) return "<UNKNOWN>";
  if (value >= 0 && value < (NumValues() - 1)) {
    return term_map_->GetTerm(value);
  }
  LOG(ERROR) << "Invalid feature value: " << value;
  return "<INVALID>";
}

string TermFrequencyMapFeature::WorkspaceName() const {
  return "term-frequency-map" + input_name_;
}

string Hyphen::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case NO_HYPHEN:
      return "NO_HYPHEN";
    case HAS_HYPHEN:
      return "HAS_HYPHEN";
  }
  return "<INVALID>";
}

FeatureValue Hyphen::ComputeValue(const Token &token) const {
  const string &word = token.word();
  // bug? if no '-', find function will return -1
  return (word.find('-') < word.length() ? HAS_HYPHEN : NO_HYPHEN);
}

// Registry for the Sentence + token index feature functions.
REGISTER_CLASS_REGISTRY("sentence+index feature function", SentenceFeature);

// Register the features defined in the header.
REGISTER_SENTENCE_IDX_FEATURE("word", Word);
REGISTER_SENTENCE_IDX_FEATURE("lcword", LowercaseWord);
REGISTER_SENTENCE_IDX_FEATURE("tag", Tag);

string Digit::GetFeatureValueName(FeatureValue value) const {
    return "";
}

FeatureValue Digit::ComputeValue(const Token &token) const {
    return 0;
}

