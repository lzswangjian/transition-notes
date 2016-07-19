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
 * from a text-based source specification of the feature extractor. Feature specification
 * parsers can be added as components. By default the feature extractor can be read
 * from an ASCII protocol buffer or in a simple feature modeling language (fml).
 *
 * A feature function is invoked with a focus. Nested feature function can be invoked with
 * another focus determined by the parent feature function.
 */
#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

#include "feature_types.h"
#include "feature.h"
#include "work_space.h"
#include "task_context.h"
#include "registry.h"
#include "utils.h"

// Use the same type for feature values as is used for predicated.
typedef int64_t Predicate;
typedef Predicate FeatureValue;

// Output feature model in FML format.
void ToFMLFunction(const FeatureFunctionDescriptor &function, string *output);
void ToFML(const FeatureFunctionDescriptor &function, string *output);

/*!
 * \brief A feature vector contains feature type and value pairs.
 */
class FeatureVector {
public:
  FeatureVector() {}

  // Adds feature type and value pair to feature vector.
  void add(FeatureType *type, FeatureValue value) {
    features_.emplace_back(type, value);
  }

  void clear() { features_.clear(); }

  int size() const { return features_.size(); }

  void reserve(int n) { features_.reserve(n); }

  // Returns feature type for an element in the feature vector.
  FeatureType *type(int index) const { return features_[index].type; }

  // Returns feature value for an element in the feature vector.
  FeatureValue value(int index) const { return features_[index].value; }

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
    * specification language. If the first line starts with #! followed by a name
    * then this name is used for instantiating a feature specification parser with
    * that name. If the language cannot be detected this way is falls back to using
    * the default language supplied.
    */
  void Parse(const string &source);

  // Returns the feature extractor descriptor.
  const FeatureExtractorDescriptor &descriptor() const { return descriptor_; }
  FeatureExtractorDescriptor *mutable_descriptor() { return &descriptor_; }

  // Returns the number of feature types in the feature extractor. Invalid
  // before Init() has been called.
  int feature_types() const { return feature_types_.size(); }

  // Returns all feature types names used by the extractor. The names are 
  // added to the types_names array. Invalid before Init() has been called.
  void GetFeatureTypeNames(vector<string> *type_names) const;

  // Returns a feature type used in the extractor. Invalid before Init() has
  // been called.
  const FeatureType *feature_type(int index) const {
    return feature_types_[index];
  }

  // Returns the feature domain size of this feature extractor.
  // N.B. The way that domain size is calculated is, for some, unintuitive. It
  // is the largest domain size of any feature type.
  FeatureValue GetDomainSize() const;

protected:
  // Initializes the feature types used by the extractor. Called from
  // FeatureExtractor<>::Init();
  void InitializeFeatureTypes();

private:
  // Initializes the top-level feature functions.
  virtual void InitializeFeatureFunctions() = 0;

  // Returns all feature types used by the extractor. The feature types are
  // added to the result array.
  virtual void GetFeatureTypes(vector<FeatureType *> *types) const = 0;

  FeatureExtractorDescriptor descriptor_;

  // All feature types used by the feature extractor. The collection of all the feature
  // types describes the feature space of the feature set produced by the feature extractor.
  vector<FeatureType *> feature_types_;
};


/*!
 * \brief The generic feature function is the type-independent part of a feature
 * function. Each feature function is associated with the descriptor that it is
 * instantiated from. The feature types associated with this feature function
 * will be established by the time FeatureExtractor<>::Init() completes.
 */
class GenericFeatureFunction {
public:
  // A feature value that represents the absence of a value.
  static constexpr FeatureValue kNone = -1;

  GenericFeatureFunction();
  virtual ~GenericFeatureFunction();

  // Sets up the feature function. N.B.: FeatureTypes of nested functions are not
  // guaranteed to be available until Init().
  virtual void Setup(TaskContext *context) {}

  virtual void Init(TaskContext *context) {}

  virtual void RequestWorkspaces(WorkspaceRegistry *registry) {}

  virtual void GetFeatureTypes(vector<FeatureType *> *types) const;

  virtual FeatureType *GetFeatureType() const;

  virtual const char *RegistryName() const = 0;

  string GetParameter(const string &name) const;
  int GetIntParameter(const string &name, int default_value) const;

  // Returns the FML function description for the feature function, i.e. the 
  // name and parameters without the nested features.
  string FunctionName() const {
    string output;
    ToFMLFunction(*descriptor_, &output);
    return output;
  }

  // Returns the prefix for nested feature functions.
  string SubPrefix() const {
    return prefix_.empty() ? FunctionName() : prefix_ + "." + FunctionName();
  }

  // Returns/sets the feature extractor this function belongs to.
  GenericFeatureExtractor *extractor() const { return extractor_; }
  void set_extractor(GenericFeatureExtractor *extractor) {
    extractor_ = extractor;
  }

  // Returns/sets the feature function descriptor.
  FeatureFunctionDescriptor *descriptor() const { return descriptor_; }
  void set_descriptor(FeatureFunctionDescriptor *descriptor) {
    descriptor_ = descriptor;
  }

  // Returns a descriptive name for the feature function. The name is taken from
  // the descriptor for the feature function.
  string name() const {
    string output;
    if (descriptor_->name().empty()) {
      if (!prefix_.empty()) {
        output.append(prefix_);
        output.append(".");
      }
      ToFML(*descriptor_, &output);
    } else {
      output = descriptor_->name();
    }
    return output;
  }

  // Returns the argument from the feature function descriptor. It defaults to
  // 0 if the argument has not been specified.
  int argument() const {
    return descriptor_->has_argument() ? descriptor_->argument() : 0;
  }

  const string &prefix() const { return prefix_; }
  void set_prefix(const string &prefix) { prefix_ = prefix; }

protected:
  // Returns the feature type for single-type feature functions.
  FeatureType *feature_type() const { return feature_type_; }

  void set_feature_type(FeatureType *feature_type) {
    CHECK(feature_type_ == nullptr);
    feature_type_ = feature_type;
  }
  
  private:
  // Feature extractor this feature function belongs to. Not owned.
    GenericFeatureExtractor *extractor_ = nullptr;

    // Descriptor for feature function. Not owned.
    FeatureFunctionDescriptor *descriptor_ = nullptr;

    /*!
     * \brief Feature type for features produced by this feature function.
     * If the feature function produces features of multiple feature types this
     * is null and feature function must return it's feature types in 
     * GetFeatureTypes(). Owned.
     */
    FeatureType *feature_type_ = nullptr;

    // Prefix used for sub-feature types of this function.
    string prefix_;
};

/*!
 * \brief Feature function that can extract features from an object. Templated on
 * two type arguments.
 *
 * OBJ: The "object" from which features are extracted; e.g., a sentence. This 
 *      should be a plain type, rather than a reference or pointer.
 *
 * ARGS: A set of 0 or more types that are used to "index" into some part of the
 *       object that should be extracted, e.g. an int token index for a sentence
 *       object. This should not be a reference type.
 */
template<class OBJ, class ...ARGS>
class FeatureFunction : public GenericFeatureFunction,
                        public RegisterableClass< FeatureFunction<OBJ, ARGS...> > {
  public:
    using Self = FeatureFunction<OBJ, ARGS...>;

    // Preprocesses the object. This will be called prior to calling Evaluate()
    // or Compute() on that object.
    virtual void Preprocess(WorkspaceSet *workspaces, OBJ *object) const {}

    // Appends features computed from the object and focus to the result.
    // The default implementation delegates to Compute(), adding a single
    // value if available. Multi-valued feature functions must override
    // this method.
    virtual void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
        ARGS... args, FeatureVector *result) const {
      FeatureValue value = Compute(workspaces, object, args..., result);
      if (value != kNone) result->add(feature_type(), value);
    }

    // Returns a feature value computed from the object and focus, or kNone if no
    // value is computed. Single-valued feature functions only need to override this
    // method.
    virtual FeatureValue Compute(const WorkspaceSet &workspaces,
        const OBJ &object, ARGS... args, const FeatureVector *fv) const {
      return kNone;
    }

  // Instantiates a new feature function in a feature extractor from a feature
  // descriptor.
  static Self *Instantiate(GenericFeatureExtractor *extractor,
                           FeatureFunctionDescriptor *fd,
                           const string &prefix) {
    Self *f = Self::Create(fd->type());
    f->set_extractor(extractor);
    f->set_descriptor(fd);
    f->set_prefix(prefix);
    return f;
  }

  // Returns the name of the registry for the feature function.
  const char *RegistryName() const override {
    return Self::registry()->name;
  }

private:
  // Special feature function class for resolving variable references. The type
  // of the feature function is used for resolving the variable reference.
  class Reference;
};

/*!
 * \brief Base class for features with nested feature functions. The nested functions
 * are of type NES, which may be different from the type of the parent function.
 */
template<class NES, class OBJ, class ...ARGS>
class NestedFeatureFunction : public FeatureFunction<OBJ, ARGS...> {
public:
  using Parent = NestedFeatureFunction<NES, OBJ, ARGS...>;

  ~NestedFeatureFunction() override {
    utils::STLDeleteElements(&nested_);
  }

  // By default, just appends the nested feature types.
  void GetFeatureTypes(vector<FeatureType *> *types) const override {
    CHECK(!this->nested().empty())
      << "Nested features require nested features to be defined.";
    for (auto *function : nested_) function->GetFeatureTypes(types);
  }

  void Setup(TaskContext *context) override {
  }

  virtual void SetupNested(TaskContext *context) {}

  void Init(TaskContext *context) override {
    for (auto *function : nested_) function->Init(context);
    InitNested(context);
  }

  // Initializes this NestedFeatureFunction specifically.
  virtual void InitNested(TaskContext *context) {}

  // Gets all the workspaces needed for the nested functions.
  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    for (auto *function : nested_) function->RequestWorkspaces(registry);
  }

  // Returns the list of nested feature functions.
  const vector<NES *> &nested() const { return nested_; }

  // Instantiates nested feature functions for a feature function. Creates
  // and initializes one feature function for each sub-descriptor in the
  // feature descriptor.
  static void CreateNested(GenericFeatureExtractor *extractor,
      FeatureFunctionDescriptor *fd,
      vector<NES *> *functions,
      const string &prefix) {
    for (int i = 0; i < fd->feature_size(); ++i) {
      FeatureFunctionDescriptor *sub = fd->mutable_descriptor(i);
      NES *f = NES::Instantiate(extractor, sub, prefix);
      functions->push_back(f);
    }
  }

protected:
  // The nested feature functions, if any, in order of declaration 
  // in the feature descriptor. Owned.
  vector<NES *> nested_;
};

template<class OBJ, class ...ARGS>
class MetaFeatureFunction : public NestedFeatureFunction<
    FeatureFunction<OBJ, ARGS...>, OBJ, ARGS...> {
public:
    void Preprocess(WorkspaceSet *workspaces, OBJ *object) const override {
      for (auto *function : this->nested_) {
        function->Preprocess(workspaces, object);
      }
    }
};

/*!
 * \brief Template for a special type of locator:
 * the locator of type FeatureFunction<OBJ, ARGS...> calls nested functions
 * of type FeatureFunction<OBJ, IDX, ARGS...>, where the derived class DER
 * is responsible for translating by providing the following:
 *
 * // Gets the new additional focus.
 * IDX GetFocus(const WorkspaceSet &workspaces, const OBJ &object);
 *
 * This is useful to e.g. add a token focus to a parser state based on
 * some desired property of that state.
 */
template<class DER, class OBJ, class IDX, class ...ARGS>
class FeatureAddFocusLocator : public NestedFeatureFunction<
    FeatureFunction<OBJ, IDX, ARGS...>, OBJ, ARGS...> {
public:
  void Preprocess(WorkspaceSet *workspaces, OBJ *object) const override {
    for (auto *function : this->nested_) {
      function->Preprocess(workspaces, object);
    }
  }

  void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
      ARGS... args, FeatureVector *result) const override {
    IDX focus = static_cast<const DER *>(this)->GetFocus(
        workspaces, object, args...);
    for (auto *function : this->nested_) {
      function->Evaluate(workspaces, object, focus, args..., result);
    }
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
      ARGS... args, const FeatureVector *result) const override {
    IDX focus = static_cast<const DER *>(this)->GetFocus(
        workspaces, object, args...);
    return this->nested()[0]->Compute(
        workspaces, object, focus, args..., result);
  }
};

/*!
 * \brief CRTP feature locator class. 
 * This is a meta feature that modifies ARGS
 * and then calls the nested feature functions with the modified ARGS.
 */
template<class DER, class OBJ, class ...ARGS>
class FeatureLocator : public MetaFeatureFunction<OBJ, ARGS...> {
  public:
    void GetFeatureTypes(vector<FeatureType *> *types) const override {
      CHECK(this->feature_type() == nullptr)
        << "FeatureLocators should not haven an intrinsic type.";
      MetaFeatureFunction<OBJ, ARGS...>::GetFeatureTypes(types);
    }

  void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
      ARGS... args, FeatureVector *result) const override {
    static_cast<const DER *>(this)->UpdateArgs(workspaces, object, &args...);
    for (auto *function : this->nested_) {
      function->Evaluate(workspaces, object, args..., result);
    }
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
      ARGS... args, const FeatureVector *result) const override {
    static_cast<const DER *>(this)->UpdateArgs(workspaces, object, &args...);
    return this->nested()[0]->Compute(
        workspaces, object, args..., result);
  }
};

/*!
 * \brief Feature extractor for extracting features from objects of a certain class.
 * Template type parameters are as defined for FeatureFunction.
 */
template<class OBJ, class ...ARGS>
class FeatureExtractor : public GenericFeatureExtractor {
public:
  typedef FeatureFunction<OBJ, ARGS...> Function;
  typedef FeatureExtractor<OBJ, ARGS...> Self;

  // Feature locator type for the feature extractor.
  template<class DER>
  using Locator = FeatureLocator<DER, OBJ, ARGS...>;

  // Initializes feature extractor.
  FeatureExtractor() {}
  ~FeatureExtractor() { utils::STLDeleteElements(&functions_); }

  void Setup(TaskContext *context) {
    for (Function *function : functions_) function->Setup(context);
  }

  void Init(TaskContext *context) {
    for (Function *function : functions_) function->Init(context);
    this->InitializeFeatureTypes();
  }

  void RequestWorkspaces(WorkspaceRegistry *registry) {
    for (Function *function : functions_) function->RequestWorkspaces(registry);
  }

  void Preprocess(WorkspaceSet *workspaces, OBJ *object) const {
    for (Function *function : functions_) {
      function->Preprocess(workspaces, object);
    }
  }

  void ExtractFeatures(const WorkspaceSet &workspaces, const OBJ &object,
                       ARGS... args, FeatureVector *result) const {
    result->reserve(functions_.size());

    // Extract features.
    for (int i = 0; i < functions_.size(); ++i) {
      functions_[i]->Evaluate(workspaces, object, args..., result);
    }
  }

private:
  // Creates and initializes all feature functions in the feature extractor.
  void InitializeFeatureFunctions() override {
    // Create all top-level feature functions.
    for (int i = 0; i < descriptor().feature_size(); ++i) {
      FeatureFunctionDescriptor *fd = mutable_descriptor()->mutable_feature(i);
      Function *function = Function::Instantiate(this, fd, "");
      functions_.push_back(function);
    }
  }

  // Collect all feature types used in the feature extractor.
  void GetFeatureTypes(vector<FeatureType *> *types) const override {
    for (int i = 0; i < functions_.size(); ++i) {
      functions_[i]->GetFeatureTypes(types);
    }
  }

  // Top-level feature functions (and variables) in the feature extractor.
  // Owned.
  vector<Function *> functions_;
};

#define REGISTER_FEATURE_FUNCTION(base, name, component) \
  REGISTER_CLASS_COMPONENT(base, name, component)

#endif
