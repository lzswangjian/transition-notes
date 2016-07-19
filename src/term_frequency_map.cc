#include "term_frequency_map.h"

int TermFrequencyMap::Increment(const string &term) {
  CHECK_EQ(term_index_.size(), term_data_.size());
  const TermIndex::const_iterator it = term_index_.find(term);
  if (it != term_index_.end()) {
    // Increment the existing term.
    pair<string, int64_t> &data = term_data_[it->second];
    CHECK_EQ(term, data.first);
    ++(data.second);
    return it->second;
  } else {
    // Add a new term.
    const int index = term_index_.size();
    CHECK_LT(index, std::numeric_limits<int32_t>::max());
    term_index_[term] = index;
    term_data_.push_back(pair<string, int64_t>(term, 1));
    return index;
  }
}

void TermFrequencyMap::Clear() {
  term_index_.clear();
  term_data_.clear();
}

void TermFrequencyMap::Load(const string &filename, int min_frequency, int max_num_terms) {
  Clear();

  // If max_num_terms is non-positive, replace it with INT_MAX.
  if (max_num_terms <= 0) max_num_terms = std::numeric_limits<int>::max();

  ifstream m_file(filename.c_str());
  if (!m_file.is_open()) {
    LOG(FATAL) << "Open file [ " << filename << " ] failed.";
  }

  string line;
  // Read head.
  std::getline(m_file, line);
  int32_t total = -1;
  CHECK(utils::ParseInt32(line.c_str(), &total));
  CHECK_GE(total, 0);

  int64_t last_frequency = -1;
  for (int i = 0; i < total && i < max_num_terms; ++i) {
    std::getline(m_file, line);
    vector<string> elements = utils::Split(line, ' ');
    CHECK_EQ(2, elements.size());
    CHECK(!elements[0].empty());
    CHECK(!elements[1].empty());
    int64_t frequency = 0;
    CHECK(utils::ParseInt64(elements[1].c_str(), &frequency));
    CHECK_GT(frequency, 0);
    const string &term = elements[0];

    // Check frequency sorting (descending order).
    if (i > 0) CHECK_GE(last_frequency, frequency);
    last_frequency = frequency;

    // Ignore low-frequency items.
    if (frequency < min_frequency) continue;

    // Check uniqueness of the mapped terms.
    CHECK(term_index_.find(term) == term_index_.end())
      << "File " << filename << " has duplicate term: " << term;

    // Assign the next avaiable index.
    const int index = term_index_.size();
    term_index_[term] = index;
    term_data_.push_back(pair<string, int64_t>(term, frequency));
  }
  m_file.close();
  CHECK_EQ(term_index_.size(), term_data_.size());
  LOG(INFO) << "Loaded " << term_index_.size() << " terms from " << filename << ".";
}

struct TermFrequencyMap::SortByFrequencyThenTerm {
  // Return a > b to sort in descending order of frequency; otherwise,
  // lexicographic sort on term.
  bool operator()(const pair<string, int64_t> &a,
      const pair<string, int64_t> &b) const {
    return (a.second > b.second || (a.second == b.second && a.first < b.first));
  }
};

void TermFrequencyMap::Save(const string &filename) const {
  CHECK_EQ(term_index_.size(), term_data_.size());

  // Copy and sort the term data.
  vector<pair<string, int64_t> > sorted_data(term_data_);
  std::sort(sorted_data.begin(), sorted_data.end(), SortByFrequencyThenTerm());

  // Write the number of terms.
  ofstream m_file(filename.c_str());
  if (!m_file.is_open()) {
    LOG(FATAL) << "Open file [ " << filename << " failed";
  }
  // Header
  const int32_t num_terms = term_index_.size();
  m_file << num_terms << endl;

  for (size_t i = 0; i < sorted_data.size(); ++i) {
    if (i > 0) CHECK_GE(sorted_data[i-1].second, sorted_data[i].second);
    m_file << sorted_data[i].first << " " << sorted_data[i].second << endl;
  }
  m_file.close();
  LOG(INFO) << "Saved " << term_index_.size() << " terms to " << filename << ".";
}

string TermFrequencyMap::ToString() const {
  string str;
  TermIndex::const_iterator it = term_index_.begin();
  for (; it != term_index_.end(); ++it) {
    str += it->first + ":";
    str += utils::Printf(it->second) + "\n";
  }
  return str;
}
