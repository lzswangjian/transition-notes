/*!
 * \brief Features that opearte on Sentence objects. Most features
 * are defined in this header so they may be re-used via composition
 * into other more advanced feature classes.
 */

#ifndef SENTENCE_FEATURES_H_
#define SENTENCE_FEATURES_H_

#include "../lexicon/affix.h"
#include "feature_extractor.h"
#include "feature_types.h"
#include "../sentence.h"
#include "../utils/task_context.h"
#include "../utils/work_space.h"
#include "../utils/registry.h"
#include "../lexicon/term_frequency_map.h"


// Feature function for any component that processes Sentences, whose
// focus if a token index into the sentence.
typedef FeatureFunction<Sentence, int> SentenceFeature;

// Alias for Locator type features that take (Sentence, int) signatures
// and call other (Sentence, int) features.
template <class DER>
using Locator = FeatureLocator<DER, Sentence, int>;

class TokenLookupFeature : public SentenceFeature {
  public:
    void Init(TaskContext *context) override {
      set_feature_type(new ResourceBasedFeatureType<TokenLookupFeature>(
            name(), this, {{NumValues(), "<OUTSIDE>"}}));
    }

    virtual FeatureValue ComputeValue(const Token &token) const = 0;

    virtual int64_t NumValues() const = 0;

    virtual string GetFeatureValueName(FeatureValue value) const = 0;

    virtual string WorkspaceName() const = 0;

    void Preprocess(WorkspaceSet *workspaces,
        Sentence *sentence) const override {
      if (workspaces->Has<VectorIntWorkspace>(workspace_)) return;
      VectorIntWorkspace *workspace = new VectorIntWorkspace(
          sentence->token_size());
      for (int i = 0; i < sentence->token_size(); ++i) {
        const int value = ComputeValue(sentence->token(i));
        workspace->set_element(i, value);
      }
      workspaces->Set<VectorIntWorkspace>(workspace_, workspace);
    }

    // Requests a vector of int's to store in the workspace registry.
    void RequestWorkspaces(WorkspaceRegistry *registry) override {
      workspace_ = registry->Request<VectorIntWorkspace>(WorkspaceName());
    }

    // Returns the precomputed value, or NumValues() for features outside 
    // the sentence.
    FeatureValue Compute(const WorkspaceSet &workspaces,
        const Sentence &sentence, int focus, const FeatureVector *result) const override {
      if (focus < 0 || focus >= sentence.token_size()) return NumValues();
      return workspaces.Get<VectorIntWorkspace>(workspace_).element(focus);
    }

  private:
    int workspace_;
};

// Lookup feature that uses a TermFrequencyMap to store a string->int mapping.
class TermFrequencyMapFeature : public TokenLookupFeature {
public:
    explicit TermFrequencyMapFeature(const string &input_name)
        : input_name_(input_name), min_freq_(0), max_num_terms_(0) {}
    ~TermFrequencyMapFeature() override;

    // Requests the input map as a resource.
    void Setup(TaskContext *context) override;

    void Init(TaskContext *context) override;

    virtual int64_t NumValues() const override { return term_map_->Size() + 1; }

    // Special value for strings not in the map.
    FeatureValue UnknownValue() const { return term_map_->Size(); }

    string GetFeatureValueName(FeatureValue value) const override;

    string WorkspaceName() const override;

protected:
    const TermFrequencyMap &term_map() const { return *term_map_; }

private:
    // Not owned.
    const TermFrequencyMap *term_map_ = nullptr;

    string input_name_;

    // Filename of the underlying resource.
    string file_name_;

    // Minimum frequency for term map.
    int min_freq_;

    // Maximum number of terms for term map.
    int max_num_terms_;
};

class Word : public TermFrequencyMapFeature {
  public:
    Word() : TermFrequencyMapFeature("word-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      string form = token.word();
      return term_map().LookupIndex(form, UnknownValue());
    }
};

class LowercaseWord : public TermFrequencyMapFeature {
  public:
    LowercaseWord() : TermFrequencyMapFeature("lc-word-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      const string lcword = utils::Lowercase(token.word());
      return term_map().LookupIndex(lcword, UnknownValue());
    }
};

class Tag : public TermFrequencyMapFeature {
  public:
    Tag() : TermFrequencyMapFeature("tag-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      return term_map().LookupIndex(token.tag(), UnknownValue());
    }
};

class Label : public TermFrequencyMapFeature {
  public:
    Label() : TermFrequencyMapFeature("label-map") {}

    FeatureValue ComputeValue(const Token &token) const override {
      return term_map().LookupIndex(token.label(), UnknownValue());
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
      return name_ + ":" + utils::Printf(cardinality_);
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
