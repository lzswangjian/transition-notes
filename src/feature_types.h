// Common feature types for parser components.

#ifndef FEATURE_TYPES_H_
#define FEATURE_TYPES_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include "utils.h"

// Use the same type for feature values as is used for predicated.
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

  // Converts a feature value to a name.
  virtual string GetFeatureValueName(FeatureValue value) const = 0;

  // Returns the size of the feature values domain.
  virtual int64_t GetDomainSize() const = 0;

  // Returns the feature type name.
  const string &name() const { return name_; }

  Predicate base() const { return base_; }

  void set_base(Predicate base) { base_ = base; }

private:
  // Feature type name.
  string name_;

  // "Base" feature value: i.e. a "slot" in a global ordering of feature.
  Predicate base_;
};

/*!
 * \brief Template generic resource based feature type. This feature type
 * delegates look up of feature value names to an unknown resource class,
 * which is not owned. Optionally, this type can also store a mapping of 
 * extra values which are not in the resource.
 *
 * Note: this class assumes that Resource->GetFeatureValueName() will return
 * successfully for values ONLY in the range [0, Resource->NumValues()).
 * Any feature value not in the extra value map and not in the above range of
 * Resource will result in a ERROR and return of "<INVALID>".
 */
template<class Resource>
class ResourceBasedFeatureType : public FeatureType {
public:
    ResourceBasedFeatureType(const string &name, const Resource *resource,
                             const map<FeatureValue, string> &values)
        : FeatureType(name), resource_(resource), values_(values) {

    }

    string GetFeatureValueName(FeatureValue value) const override {

    }

    int64_t GetDomainSize() const override {
        return max_value_ + 1;
    }


protected:
    // Shared resource. Not owned.
  const Resource *resource_ = nullptr;

  FeatureValue max_value_;

  map<FeatureValue, string> values_;
};


/*!
 * \brief Feature type that is defined using an explicit map
 * from FeatureValue to string values. This can reduce some of the
 * boilerplate when defining features that generate enum values.
 * Example usage:
 *
 *
 */
class EnumFeatureType : public FeatureType {
public:
  EnumFeatureType(const string &name,
                  const map<FeatureValue, string> &value_names)
    : FeatureType(name), value_names_(value_names) {
    for (const auto &pair : value_names) {
      CHECK_GE(pair.first, 0) << "Invalid feature value:"
        << pair.first << ", " << pair.second;
      domain_size_ = std::max(domain_size_, pair.first + 1);
    }
  }

  // Returns the feature name for a given feature value.
  string GetFeatureValueName(FeatureValue value) const override {
    auto it = value_names_.find(value);
    if (it == value_names_.end()) {
      LOG(ERROR) << "Invalid feature value " << value << " for " << name();
      return "<INVALID>";
    } else {
      return it->second;
    }
  }

  // Returns the number of possible values for this feature type.
  // This is one greater than the largest value in the value_names map.
  FeatureValue GetDomainSize() const override { return domain_size_; }

protected:
  // Maximum possible value this feature could take.
  FeatureValue domain_size_ = 0;

  // Names of feature values.
  map<FeatureValue, string> value_names_;
};
#endif
