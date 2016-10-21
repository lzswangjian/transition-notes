//
// Created by ShengLi on 13/10/2016.
//

#ifndef SYNTAXNET_SCOREMATRIX_H_
#define SYNTAXNET_SCOREMATRIX_H_

class ScoreMatrix {
 public:
  float *data_ptr_;
  int row_;
  int col_;

  float operator()(int row, int col) {
    return data_ptr_[row * col_ + col];
  }

  float *mutable_data() { return data_ptr_; }

  const float &data() const { return *data_ptr_; }

  const int row() const { return row_; }

  const int col() const { return col_; }
};

#endif //SYNTAXNET_SCOREMATRIX_H_
