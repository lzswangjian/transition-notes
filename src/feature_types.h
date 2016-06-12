#ifndef FEATURE_TYPES_H_
#define FEATURE_TYPES_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>

typedef int64_t Predicate;
typedef Predicate FeatureValue;

/*!
 * \brief Each feature value in a feature vector has a feature type.
 * The feature type is used for converting feature type and value pairs
 * to predicate values. The feature type can also return names for feature
 * values and calculate the size of the feature value domain. The
 * FeatureType class is abstract and must be specialized for the concrete
 * feature types.
 */
class FeatureType {
public:
  // Initializes a feature type.
  explicit FeatureType(const string &name)
    : name_(name), base_(0) {}

  virtual ~FeatureType() {}

private:
  // Feature type name.
  string name_;

  // "Base" feature value: i.e. a "slot" in a global ordering of feature.
  Predicate base_;
};

/*!
 * \brief 
 */
template<class Resource>
class ResourceBasedFeatureType : public FeatureType {
protected:
  const Resource *resource_ = nullptr;

  FeatureValue max_value_;

  map<FeatureValue, string> values_;
};


/*!
 * \brief Feature type that is defined using an explicit map
 * from FeatureValue to string values.
 */
class EnumFeatureType : public FeatureType {

protected:
  // Maximum possible value this feature could take.
  FeatureValue domain_size_ = 0;

  // Names of feature values.
  map<FeatureValue, string> value_names_;
};
#endif
