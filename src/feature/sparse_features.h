#ifndef SYNTAXNET_SPARSE_FEATURES_H
#define SYNTAXNET_SPARSE_FEATURES_H

#include "../utils/utils.h"

/*!
 * \brief A sparse set of features.
 * If using SparseStringToIdTransformer, description is required and id
 * should be omitted; otherwise, id is required and description optional.
 *
 * id, weight, and description fields are all algined if present (i.e., any
 * of these that are non-empty should have the same # items). If weight is
 * omitted, 1.0 is used.
 */
class SparseFeatures {
public:
    std::vector<uint64_t> id_;
    std::vector<float> weight_;
    std::vector<std::string> description_;

    void add_id(uint64_t id) {
        id_.push_back(id);
    }

    void add_weight(float weight) {
        weight_.push_back(weight);
    }

    void add_description(const string &description) {
        description_.push_back(description);
    }

};
#endif //SYNTAXNET_SPARSE_FEATURES_H
