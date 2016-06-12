#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

typedef int64_t Predicate;
typedef Predicate FeatureValue;

class FeatureVector {
  public:
    FeatureVector() {}

  private:
    // Structure for holding feature type and value pairs.
    struct Element {
      Element() : type(NULL), value(-1) {}
      Element(FeatureType *t, FeatureValue v) : type(t), value(v) {}

      FeatureType *type;
      FeatureValue value;
    };

    // Array for storing feature vector elements.
    vector<Element> features_;
};

/*!
 * \brief The generic feature extractor is the type-independent part of a
 * feature extractor. This holds the descriptor for the feature extractor and
 * the collection of feature types used in the feature extractor. The feature
 * types are not available until FeatureExtractor<>::Init() has been called.
 */
class GenericFeatureExtractor {
  public:
    GenericFeatureExtractor();
    virtual ~GenericFeatureExtractor();

    /*!
     * \brief Initializes the feature extractor from a source representation of
     * the feature extractor. The first line is used for determining the feature
     * specification language.
     */
    void Parse(const string &source);

    // Returns the feature extractor descriptor.
    const FeatureExtractorDescriptor &decriptor() const { return descriptor_; }
    FeatureExtractorDescriptor *mutable_descriptor() { return &descriptor_; }

  private:
    virtual void InitializeFeatureFunctions() = 0;

    virtual void GetFeatureTypes(vector<FeatureType *> *types) const = 0;

    FeatureExtractorDescriptor descriptor_;
};


/*!
 * \brief The generic feature function is the type-independent part of a feature
 * function. Each feature function is associated with the descriptor that it is
 * instantiated from. The feature types associated with this feature function
 * will be established by the time FeatureExtractor<>::Init() completes.
 */
class GenericFeatureFunction {
  private:
    GenericFeatureExtractor *extractor_ = nullptr;

    FeatureExtractorDescriptor *descriptor_ = nullptr;

    /*!
     * \brief Feature type for features produced by this feature function.
     */
    FeatureType *feature_type_ = nullptr;

    // Prefix used for sub-feature types of this function.
    string prefix_;
};

template<class OBJ, class ...ARGS>
class FeatureFunction : public GenericFeatureFunction {
  public:
    using Self = FeatureFunction<OBJ, ARGS...>;

};

#endif
