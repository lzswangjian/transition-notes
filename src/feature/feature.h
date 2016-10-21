//
// Created by ShengLi on 7/12/16.
//

#ifndef SYNTAXNET_FEATURE_H_
#define SYNTAXNET_FEATURE_H_

#include "../utils/utils.h"

class Parameter {
 public:
  string name_;
  string value_;

  const string &name() const { return name_; }

  void set_name(const string &name);

  const string &value() const { return value_; }

  void set_value(const string &value);
};

/*!
 * \brief Descriptor for feature function.
 */
class FeatureFunctionDescriptor {
 public:
  // Feature function type.
  string type_;

  // Feature function name.
  string name_;

  // Default argument for feature function;
  int32_t argument_ = 0;

  // Named parameters for feature descriptor.
  vector<Parameter *> parameter_;

  // Nested sub-feature function descriptors.
  vector<FeatureFunctionDescriptor *> feature_;

  const string &type() const { return type_; }

  void set_type(const string &type);

  const string &name() const { return name_; }

  void set_name(const string &name);

  FeatureFunctionDescriptor *add_feature();

  int32_t argument() const { return argument_; }

  void set_argument(int argument);

  bool has_argument() { return true; }

  Parameter *add_parameter();

  int parameter_size() const { return parameter_.size(); }

  const Parameter &parameter(int index) const {
    return *parameter_[index];
  }

  int32_t feature_size() const { return feature_.size(); }

  FeatureFunctionDescriptor *mutable_descriptor(int index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, feature_.size());
    return feature_[index];
  }

  const FeatureFunctionDescriptor &feature(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, feature_.size());
    return *feature_[index];
  }
};

/*!
 * \brief Descriptor for feature extractor.
 */
class FeatureExtractorDescriptor {
 public:
  // Top level feature function for extractor.
  std::vector<FeatureFunctionDescriptor *> feature_;

  FeatureFunctionDescriptor *add_feature();

  int32_t feature_size() const { return feature_.size(); }

  FeatureFunctionDescriptor *mutable_feature(int index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, feature_.size());
    return feature_[index];
  }
};


#endif //SYNTAXNET_FEATURE_H_
