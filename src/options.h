#ifndef OPTIONS_H
#define OPTIONS_H

#include "base.h"

struct Options {
  string input_file_ = "/Users/Sheng/WorkSpace/transition-notes/test/dev.conll.utf8";
  string word_map_file_ = "/Users/Sheng/WorkSpace/transition-notes/word-map";
  string lc_word_map_file_ = "/Users/Sheng/WorkSpace/transition-notes/lcword-map";
  string tag_map_file_ = "/Users/Sheng/WorkSpace/transition-notes/tag-map";
  string category_map_file_ = "/Users/Sheng/WorkSpace/transition-notes/category-map";
  string label_map_file_ = "/Users/Sheng/WorkSpace/transition-notes/label-map";
};

#endif /* end of include guard: OPTIONS_H */
