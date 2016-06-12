#ifndef TERM_FREQUENCY_MAP_H_
#define TERM_FREQUENCY_MAP_H_

#include <stddef.h>
#include <string>
#include <unordered_map>
#include <vector>

class TermFrequencyMap {
  public:
    TermFrequencyMap() {}
    
    int Size() const { return term_index_.size(); }

    int LookupIndex(const string &term, int unknown) const {
    }

    const string &GetTerm(int index) const {
    }

    void Clear();

    void Load(const string &filename, int min_frequency, int max_num_terms);

    void Save(const string &filename) const;

  private:
    typedef std::unordered_map<string, int> TermIndex;

    TermIndex term_index_;
};

#endif
