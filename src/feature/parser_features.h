#ifndef PARSER_FEATURES_H_
#define PARSER_FEATURES_H_
/*!
 * \brief Sentence-based features for the transition parser.
 */
#include "feature_types.h"
#include "../parser/parser_state.h"
#include "feature_extractor.h"
#include "../utils/registry.h"

// A union used to represented discrete and continuous feature values.
union FloatFeatureValue {
public:
    explicit FloatFeatureValue(FeatureValue v) : discrete_value(v) {}

    FloatFeatureValue(int32_t i, float w) : id(i), weight(w) {}

    FeatureValue discrete_value;
    struct {
        uint32_t id;
        float weight;
    };
};

typedef FeatureFunction<ParserState> ParserFeatureFunction;

// Feature function for the transition parser based on a parser state object
// and a token index. This typically extracts information from a given token.
typedef FeatureFunction<ParserState, int> ParserIndexFeatureFunction;

// Feature extractor for the transition parser based on a parser state object.
typedef FeatureExtractor<ParserState> ParserFeatureExtractor;

// Utilities to register the two types of parser features.
#define REGISTER_PARSER_FEATURE_FUNCTION(name, component) \
  REGISTER_FEATURE_FUNCTION(ParserFeatureFunction, name, component)

#define REGISTER_PARSER_IDX_FEATURE_FUNCTION(name, component) \
  REGISTER_FEATURE_FUNCTION(ParserIndexFeatureFunction, name, component)

// Alias for locator type that takes a parser state, and produces a focus
// integer that can be used on nested ParserIndexFeature objects.
template<class DER>
using ParserLocator = FeatureAddFocusLocator<DER, ParserState, int>;

// Alias for locator type features that take (ParserState, int) signatures and
// call other ParserIndexfeatures.
template<class DER>
using ParserIndexLocator = FeatureLocator<DER, ParserState, int>;

// A simple wrapper FeatureType that adds a special "<ROOT>" type.
class RootFeatureType : public FeatureType {
public:
    RootFeatureType(const string &name, const FeatureType &wrapped_type,
                    int root_value);

    // Returns the feature value name, but with the special "<ROOT>" value.
    string GetFeatureValueName(FeatureValue value) const override;

    FeatureValue GetDomainSize() const override;

private:
    const FeatureType &wrapped_type_;

    // The reserved root value.
    int root_value_;
};

// Simple feature function that wraps a Sentence based feature function.
// It adds a "<ROOT>" feature value that is triggered whenever the focus is
// the special root token.
template<class F>
class ParserSentenceFeatureFunction : public ParserIndexFeatureFunction {
public:
    void Setup(TaskContext *context) override {
        this->feature_.set_descriptor(this->descriptor());
        this->feature_.set_prefix(this->prefix());
        this->feature_.set_extractor(this->extractor());
        feature_.Setup(context);
    }

    void Init(TaskContext *context) override {
        feature_.Init(context);
        num_base_values_ = feature_.GetFeatureType()->GetDomainSize();
        num_base_values_ = feature_.GetFeatureType()->GetDomainSize();
        set_feature_type(new RootFeatureType(name(), *feature_.GetFeatureType(), RootValue()));
    }

    void RequestWorkspaces(WorkspaceRegistry *registry) override {
        feature_.RequestWorkspaces(registry);
    }

    void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
        feature_.Preprocess(workspaces, state->mutable_sentence());
    }

protected:
    // Returns the special value to represent a root token.
    FeatureValue RootValue() const { return num_base_values_; }

    // Store the number of base values from the wrapped function so compute the 
    // root value.
    int num_base_values_;

    // The wrapped feature.
    F feature_;
};

template<class F>
class BasicParserSentenceFeatureFunction :
        public ParserSentenceFeatureFunction<F> {
public:
    FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                         int focus, const FeatureVector *result) const override {
        if (focus == -1) return this->RootValue();
        return this->feature_.Compute(workspaces, state.sentence(), focus, result);
    }
};

#endif
