/*!
 * \brief Features that opearte on Sentence objects. Most features
 * are defined in this header so they may be re-used via composition
 * into other more advanced feature classes.
 */

#ifndef SENTENCE_FEATURES_H_
#define SENTENCE_FEATURES_H_

#include "affix.h"
#include "feature_extractor.h"
#include "feature_types.h"


// Feature function for any component that processes Sentences, whose
// focus if a token index into the sentence.
typedef FeatureFunction<Sentence, int> SentenceFeature;

// Alias for Locator type features that take (Sentence, int) signatures
// and call other (Sentence, int) features.
template <class DER>
using Locator = FeatureLocator<DER, Sentence, int>;

class TokenLookupFeature : public SentenceFeature {
};

// Lookup feature that uses a TermFrequencyMap to store a string->int mapping.
class TermFrequencyMapFeature : public TokenLookupFeature {
};

class Word : public TermFrequencyMapFeature {
  public:
    Word() : TermFrequencyMapFeature("word-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      string form = token.word();
      return term_map().LookupIndex(form, UnKnownValue());
    }
};

class LowercaseWord : public TermFrequencyMapFeature {
  public:
    LowercaseWord() : TermFrequencyMapFeature("lc-word-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      const string lcword = utils::Lowercase(token.word());
      return term_map().LookupIndex(lcword, UnKnownValue());
    }
};

class Tag : public TermFrequencyMapFeature {
  public:
    Tag() : TermFrequencyMapFeature("tag-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      return term_map().LookupIndex(token.tag(), UnKnownValue());
    }
};

class Label : public TermFrequencyMapFeature {
  public:
    Label() : TermFrequencyMapFeature("label-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      return term_map().LookupIndex(token.label(), UnKnownValue());
    }
};

class LexicalCategoryFeature : public TokenLookupFeature {
  public:
    LexicalCategoryFeature(const string &name, int cardinality) 
      : name_(name), cardinality_(cardinality) {}
    ~LexicalCategoryFeature() override {}

    FeatureValue NumValues() const override { return cardinality_; }

    // Returns the identifier for the workspace for this preprocessor.
    string WorkspaceName() const override {
      return tensorflow::strings::StrCat(name_, ":", cardinality_);
    }

  private:
    // Name of the category type.
    const string name_;

    // Number of values.
    const int cardinality_;
};

// Preprocessor that computes whether a word has a hyphen or not.
class Hyphen : public LexicalCategoryFeature {
  public:
    enum Category {
      NO_HYPHEN = 0,
      HAS_HYPHEN = 1,
      CARDINALITY = 2,
    };

    Hyphen() : LexicalCategoryFeature("hyphen", CARDINALITY) {}

    // Returns a string representation of the enum value.
    string GetFeatureValueName(FeatureValue value) const override;

    // Returns the category value for the token.
    FeatureValue ComputeValue(const Token &token) const override;
};

// Preprocessor that computes whether a word has a digit or not.
class Digit : public LexicalCategoryFeature {
  public:
    enum Category {
      NO_DIGIT = 0,
      SOME_DIGIT = 1,
      ALL_DIGIT = 2,
      CARDINALITY = 3,
    };

    Digit() : LexicalCategoryFeature("Digit", CARDINALITY) {}

    // Returns a string representation of the enum value.
    string GetFeatureValueName(FeatureValue value) const override;

    // Returns the category value for the token.
    FeatureValue ComputeValue(const Token &token) const override;
};


/*!
 * \brief TokenLookupPreprocessor object to compute prefixes and suffixes of words.
 * The AffixTable is stored in the SharedStore. This is very similar to the implementaton
 * of TermFrequencyMapFeature, but using an AffixTable to perform the lookups. There are
 * only two specializations, for prefixes and suffixes.
 */
class AffixTableFeature : public TokenLookupFeature {
  public:
    explicit AffixTableFeature(AffixTable::Type type);
    ~AffixTableFeature() override;

    // Requests inputs for the affix table.
    void Setup(TaskContext *context) override;

  private:
    // Size parameter for the affix table.
    int affix_length_;

    // Name of the input for the table.
    string input_name_;

    // The type of affix table (prefix or suffix).
    const AffixTable::Type type_;

    // Affix table used for indexing. This comes from the shared store, and
    // is not owned directly.
    const AffixTable *affix_table_ = nullptr;
};

class PrefixFeature : public AffixTableFeature {
  public:
    PrefixFeature() : AffixTableFeature(AffixTable::PREFIX) {}
};

class SuffixFeature : public AffixTableFeature {
  public:
    SuffixFeature() : AffixTableFeature(AffixTable::SUFFIX) {}
};

class Offset : public Locator<Offset> {
  public:
    void UpdateArgs(const WorkspaceSet &workspaces,
        const Sentence &sentence, int *focus) const {
      *focus += argument();
    }
};

typedef FeatureExtractor<Sentence, int> SentenceExtractor;

// Utility to register the sentence_instance::Feature functions.
#define REGISTER_SENTENCE_IDX_FEATURE(name, type) \
  REGISTER_FEATURE_FUNCTION(SentenceFeature, name, type)

#endif
