
#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <string>
#include <vector>
#include "../base.h"

namespace utils {

    bool ParseInt32(const char *c_str, int32_t *value);

    bool ParseInt64(const char *c_str, int64_t *value);

    bool ParseDouble(const char *c_str, double *value);

    template<typename T>
    T ParseUsing(const string &str, std::function<bool(const char *, T *)> func) {
        T value;
        CHECK(func(str.c_str(), &value)) << "Failed to convert: " << str;
        return value;
    }

    template<typename T>
    T ParseUsing(const string &str, T defval,
                 std::function<bool(const char *, T *)> func) {
        return str.empty() ? defval : ParseUsing<T>(str, func);
    }

    template<typename T>
    string Printf(T value) {
        ostringstream os;
        os << value;
        return os.str();
    }

    vector<string> Split(const string &text, char delim);

    string Join(const vector<string> &fileds, const string &delim);

    string Lowercase(const string &s);

    void NormalizeDigits(string *form);

    template<typename T>
    void STLDeleteElements(T *container) {
        if (!container) return;
        auto it = container->begin();
        while (it != container->end()) {
            auto temp = it;
            ++it;
            delete *temp;
        }
        container->clear();
    }

}

#endif /* UTILS_H */
