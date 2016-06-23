/*!
 * \brief Generic feature extractor for extracting features from objects.
 * The feature extractor can be used for extracting features from any object.
 * The feature extractor and feature function classes are template classes
 * that have to be instantiated for extracting feature from a specific object
 * type.
 *
 * A feature extractor consists of a hierarchy of feature functions. Each
 * feature function extracts one or more feature type and value pairs from the
 * object.
 *
 * The feature extractor has a modular design where new feature functions can be
 * registered as components. The feature extractor is initialized from a descriptor
 * represented by a protocol buffer. The feature extractor can also be initialized
 * from a text-based source sepcification of the feature extractor. Feature specification
 * parsers can be added as components. By default the feature extractor can be read
 * from an ASCII protocol buffer or in a simple feature modeling language (fml).
 */
#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

#include "feature_types.h"

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

/*!
 * \brief Feature function that can extract features from an object. Templated on
 * two arguments.
 *
 * OBJ: The "object" from which features are extracted; e.g., a sentence. This 
 *      should be a plain type, rather than a reference or pointer.
 *
 * ARGS: A set of 0 or more types that are used to "index" into some part of the
 *       object that should be extracted, e.g. an int token index for a sentence
 *       object. This should not be a reference type.
 */
template<class OBJ, class ...ARGS>
class FeatureFunction : public GenericFeatureFunction {
  public:
    using Self = FeatureFunction<OBJ, ARGS...>;

    // Preprocesses the object. This will be called prior to calling Evaluate()
    // or Compute() on that object.
    virtual void Preprocess(WorkspaceSet *workspaces, OBJ *object) const {}

    // Multi-valued feature
    virtual void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
        ARGS... args, FeatureVector *result) const {
    }

    // Returns a feature value computed from the object and focus, or kNone if no
    // value is computed. Single-valued feature functions only need to override this
    // method.
    virtual FeatureValue Compute(const WorkspaceSet &workspaces,
        const OBJ &object, ARGS... args, const FeatureVector *fv) const {
    }
};

/*!
 * \brief Base class for features with nested feature functions.
 */
template<class NES, class OBJ, class ...ARGS>
class NestedFeatureFunction : public FeatureFunction<OBJ, ARGS...> {
};

template<class OBJ, class ...ARGS>
class MetaFeatureFunction : public NestedFeatureFunction<
    FeatureFunction<OBJ, ARGS...>, OBJ, ARGS...> {
};

/*!
 * \brief Template for a special type of locator:
 *
 * This is useful to e.g. add a token focus to a parser state based on
 * some desired property of that state.
 */
template<class DER, class OBJ, class IDX, class ...ARGS>
class FeatureAddFocusLocator : public NestedFeatureFunction<
    FeatureFunction<OBJ, IDX, ARGS...>, OBJ, ARGS...> {
};

/*!
 * \brief CRTP feature locator class. 
 * This is a meta feature that modifies ARGS
 * and then calls the nested feature functions with the modified ARGS.
 */
template<class DER, class OBJ, class ...ARGS>
class FeatureLocator : public MetaFeatureFunction<OBJ, ARGS...> {
};

/*!
 * \brief Feature extractor for extracting features from objects of a certain class.
 */
template<class OBJ, class ...ARGS>
class FeatureExtractor : public GenericFeatureExtractor {
};

#endif
