#include "feature_extractor.h"
#include "fml_parser.h"

constexpr FeatureValue GenericFeatureFunction::kNone;

GenericFeatureExtractor::GenericFeatureExtractor() {}

GenericFeatureExtractor::~GenericFeatureExtractor() {}

void GenericFeatureExtractor::Parse(const string &source) {
  // Parse feature specification into descriptor.
  FMLParser parser;
  parser.Parse(source, mutable_descriptor());

  // Initialize feature extractor from descriptor.
  InitializeFeatureFunctions();
}

void GenericFeatureExtractor::InitializeFeatureTypes() {
  // Register all feature types.
  GetFeatureTypes(&feature_types_);

  for (size_t i = 0; i < feature_types_.size(); ++i) {
    FeatureType *ft = feature_types_[i];
    ft->set_base(i);

    // Check for feature space overflow.
    double domain_size = ft->GetDomainSize();
    if (domain_size < 0) {
      LOG(FATAL) << "Illegal domain size for feature " << ft->name()
        << domain_size;
    }
  }

  vector<string> types_names;
  GetFeatureTypeNames(&types_names);
  CHECK_EQ(feature_types_.size(), types_names.size());
}

void GenericFeatureExtractor::GetFeatureTypeNames(vector<string> *type_names) const {
  for (size_t i = 0; i < feature_types_.size(); ++i) {
    FeatureType *ft = feature_types_[i];
    type_names->push_back(ft->name());
  }
}

FeatureValue GenericFeatureExtractor::GetDomainSize() const {
  // Domain size of the set of features is equal to:
  // [largest domain size of any feature types] * [number of feature types]
  FeatureValue max_feature_type_dsize = 0;
  for (size_t i = 0; i < feature_types_.size(); ++i) {
    FeatureType *ft = feature_types_[i];
    const FeatureValue feature_type_dsize = ft->GetDomainSize();
    if (feature_type_dsize > max_feature_type_dsize) {
      max_feature_type_dsize = feature_type_dsize;
    }
  }

  return max_feature_type_dsize;
}

// Below is Implementation about GenericFeatureFunction

GenericFeatureFunction::GenericFeatureFunction() {}

GenericFeatureFunction::~GenericFeatureFunction() {
  delete feature_type_;
}

int GenericFeatureFunction::GetIntParameter(const string &name, int default_value) const {
  string value = GetParameter(name);
  return utils::ParseUsing<int>(value, default_value, utils::ParseInt32);
}

void GenericFeatureFunction::GetFeatureTypes(vector<FeatureType *> *types) const {
  if (feature_type_ != nullptr) {
    types->push_back(feature_type_);
  }
}

FeatureType *GenericFeatureFunction::GetFeatureType() const {
  // If a single feature type has been registered return it.
  if (feature_type_ != nullptr) return feature_type_;

  // Get feature types for function.
  vector<FeatureType *> types;
  GetFeatureTypes(&types);

  // If there is exactly one feature type return this, else return null.
  if (types.size() == 1) return types[0];
  return nullptr;
}

string GenericFeatureFunction::GetParameter(const string &name) const {
    for (int i = 0; i < descriptor_->parameter_size(); ++i) {
        if (descriptor_->parameter(i).name() == name) {
            return descriptor_->parameter(i).value();
        }
    }
    return "";
}


