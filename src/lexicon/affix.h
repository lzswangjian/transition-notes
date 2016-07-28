#ifndef AFFIX_H_
#define AFFIX_H_

#include "../utils/utils.h"

/*!
 * \brief An affix represents a prefix or suffix of a word of a certain length.
 * Each affix has a unique id and a textual form. An affix also has a pointer to
 * the affix that is one character shorter. This creates a chain of affixes that 
 * are successively shorter.
 */
class Affix {
private:
    friend class AffixTable;

    Affix(int id, const char *form, int length)
            : id_(id), length_(length), form_(form), shorter_(NULL), next_(NULL) {}

private:
    // Affix id.
    int id_;

    // Length (in characters) of affix.
    int length_;

    // Text form of affix.
    string form_;

    // Pointer to affix that is one character shorter.
    Affix *shorter_;

    // Next affix in bucket chain.
    Affix *next_;
};


class AffixTable {
public:
    // Affix table type.
    enum Type {
        PREFIX, SUFFIX
    };

private:
    // Adds a new affix to table.
    Affix *AddNewAffix(const string &form, int length);

    // Finds existing affix in table.
    Affix *find(const string &form) const;

    // Affix type (prefix or suffix).
    Type type_;

    // Maximum length of affix.
    int max_length_;

    // Index from affix ids to affix items.
    vector<Affix *> affixes_;

    // Buckets for word-to-affix hash map.
    vector<Affix *> buckets_;
};

#endif
