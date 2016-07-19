#include "utils.h"

namespace utils {

bool ParseInt32(const char *c_str, int32_t *value) {
  char *temp;
  *value = strtol(c_str, &temp, 0);  // NOLINT
  return (*temp == '\0');
}

bool ParseInt64(const char *c_str, int64_t *value) {
  char *temp;
  *value = strtol(c_str, &temp, 0);  // NOLINT
  return (*temp == '\0');
}

bool ParseDouble(const char *c_str, double *value) {
  char *temp;
  *value = strtod(c_str, &temp);
  return (*temp == '\0');
}

vector<string> Split(const string &text, char delim) {
  vector<string> result;
  int token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; ++i) {
      if ((i == text.size()) || (text[i] == delim)) {
        result.push_back(string(text.data() + token_start, i - token_start));
        token_start = i + 1;
      }
    }
  }
  return result;
}

string Join(const vector<string> &fields, const string &delim) {
  if (fields.size() == 0) return "";
  string ret = fields[0];
  for (size_t i = 1; i < fields.size(); ++i) {
    ret.append(delim);
    ret.append(fields[i]);
  }
  return ret;
}

string Lowercase(const string &s) {
  string result(s.data(), s.size());
  for (char &c : result) {
    c = tolower(c);
  }
  return result;
}

void NormalizeDigits(string *form) {
  for (size_t i = 0; i < form->size(); ++i) {
    if ((*form)[i] >= '0' && (*form)[i] <= '9') (*form)[i] = '9';
  }
}

} /* namespace utils */
