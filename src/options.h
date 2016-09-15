#ifndef OPTIONS_H
#define OPTIONS_H

#include "base.h"

struct Options {
  string input_file_ = "test/train.conll.utf8";
  string word_map_file_ = "word-map";
  string lc_word_map_file_ = "lcword-map";
  string tag_map_file_ = "tag-map";
  string category_map_file_ = "category-map";
  string label_map_file_ = "label-map";
};

#endif /* end of include guard: OPTIONS_H */
