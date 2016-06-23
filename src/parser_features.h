#ifndef PARSER_FEATURES_H_
#define PARSER_FEATURES_H_
/*!
 * \brief Sentence-based features for the transition parser.
 */

// A union used to represented discrete and continuous feature values.
union FloatFeatureValue {
  public:
    explicit FloatFeatureValue(FeatureValue v) ; discrete_value(v) {}
    FloatFeatureValue(uint32 i, float w) : id(i), weight(w) {}
    FeatureValue discrete_value;
    struct {
      uint32 id;
      float weight;
    };
};

typedef FeatureFunction<ParserState> ParserFeatureFunction;

// Feature function for the transition parser based on a parser state object
// and a token index. This typically extracts information from a given token.
typedef FeatureFunction<ParserState, int> ParserIndexFeatureFunction;

// Feature extractor for the transiton parser based on a parser state objec.
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
using ParserIndexFeatureLocator = FeatureLocator<DER, ParserState, int>;

// A simple wrapper FeatureType that adds a special "<ROOT>" type.
class RootFeatureType : public FeatureType {
public:
  RootFeatureType(const string &name, const FeatureType &wrapped_type,
                  int root_value);

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
#endif
