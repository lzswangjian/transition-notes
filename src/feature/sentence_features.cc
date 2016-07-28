#include "sentence_features.h"
#include "../utils/shared_store.h"

TermFrequencyMapFeature::~TermFrequencyMapFeature() {
    if (term_map_ != nullptr) {
        SharedStore::Release(term_map_);
        term_map_ = nullptr;
    }
}

void TermFrequencyMapFeature::Setup(TaskContext *context) {
  TokenLookupFeature::Setup(context);
  context->GetInput(input_name_, "text", "");
}

void TermFrequencyMapFeature::Init(TaskContext *context) {
    min_freq_ = GetIntParameter("min-freq", 0);
    max_num_terms_ = GetIntParameter("max-num-terms", 0);
    file_name_ = context->InputFile(*context->GetInput(input_name_));
    term_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
            file_name_, min_freq_, max_num_terms_);
    TokenLookupFeature::Init(context);
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
  return SharedStoreUtils::CreateDefaultName("term-frequency-map", input_name_,
                                             min_freq_, max_num_terms_);
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

string Digit::GetFeatureValueName(FeatureValue value) const {
    return "";
}

FeatureValue Digit::ComputeValue(const Token &token) const {
    return 0;
}

// Register the features defined in the header.
REGISTER_SENTENCE_IDX_FEATURE("word", Word);
REGISTER_SENTENCE_IDX_FEATURE("lcword", LowercaseWord);
REGISTER_SENTENCE_IDX_FEATURE("tag", Tag);
