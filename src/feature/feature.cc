//
// Created by ShengLi on 7/12/16.
//

#include "feature.h"

void FeatureFunctionDescriptor::set_type(const std::string &type) {
    type_ = type;
}

void FeatureFunctionDescriptor::set_name(const std::string &name) {
    name_ = name;
}

FeatureFunctionDescriptor *FeatureFunctionDescriptor::add_feature() {
    FeatureFunctionDescriptor * sub = new FeatureFunctionDescriptor();
    feature_.push_back(sub);
    return sub;
}

void FeatureFunctionDescriptor::set_argument(int argument) {
    argument_ = argument;
}

Parameter *FeatureFunctionDescriptor::add_parameter() {
    Parameter *par = new Parameter();
    parameter_.push_back(par);
    return par;
}


FeatureFunctionDescriptor *FeatureExtractorDescriptor::add_feature() {
    FeatureFunctionDescriptor *feat = new FeatureFunctionDescriptor();
    feature_.push_back(feat);
    return feat;
}

void Parameter::set_name(const std::string &name) {
    name_ = name;
}

void Parameter::set_value(const std::string &value) {
    value_ = value;
}

