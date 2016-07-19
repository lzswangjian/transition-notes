#ifndef TERM_FREQUENCY_MAP_H_
#define TERM_FREQUENCY_MAP_H_

#include <stddef.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include "utils.h"

/*!
 * \brief A mapping from strings to frequencies with save and load
 * functionality.
 */
class TermFrequencyMap {
  public:
    TermFrequencyMap() {}

    TermFrequencyMap(const string &file, int min_frequency, int max_num_terms) {
      Load(file, min_frequency, max_num_terms);
    }
    
    int Size() const { return term_index_.size(); }

    // Returns the index associated with the given term. If the
    // term does not exist, the unknown index is returned instead.
    int LookupIndex(const string &term, int unknown) const {
      const TermIndex::const_iterator it = term_index_.find(term);
      return (it != term_index_.end() ? it->second : unknown);
    }

    // Returns the term associated with the given index.
    const string &GetTerm(int index) const {
      return term_data_[index].first;
    }

    // Increases the frequency of the given term by 1, creating a
    // new entry if necessary, and returns the index of the term.
    int Increment(const string &term);

    void Clear();

    void Load(const string &filename, int min_frequency, int max_num_terms);

    void Save(const string &filename) const;

    string ToString() const;

  private:
    // Hashtable for term-to-index mapping.
    typedef std::unordered_map<string, int> TermIndex;

    // Sorting functor for term data.
    struct SortByFrequencyThenTerm;

    TermIndex term_index_;

    vector<pair<string, int64_t> > term_data_;
};

class TagToCategoryMap {
  public:
    TagToCategoryMap() {}
    ~TagToCategoryMap() {}

    explicit TagToCategoryMap(const string &filename){}

    void SetCategory(const string &tag, const string &category) {}

    const string &GetCategory(const string &tag) const { return ""; }

    void Save(const string &filename) const {}

  private:
    map<string, string> tag_to_category_;
};
#endif
