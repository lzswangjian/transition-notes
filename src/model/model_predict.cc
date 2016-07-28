#include <stdio.h>

#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>

#include "../utils/utils.h"
#include "../utils/task_context.h"

/*!
 * \brief Score matrix for output
 */
class Matrix {
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

/*!
 * \brief BufferFile is used to load mxnet symbol json and parameters.
 */
class BufferFile {
public:
    explicit BufferFile(const string &file_path) : file_path_(file_path) {
        ifstream ifs(file_path_.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            LOG(FATAL) << "Can't open file [ " << file_path_ << "]";
        }
        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }

    int GetLength() { return length_; }

    char *GetBuffer() { return buffer_; }

private:
    string file_path_;
    int length_;
    char *buffer_;
};


class Model {
public:
    Model() : batch_size_(1) {

    }

    Model(int batch_size) : batch_size_(batch_size) {
        num_input_nodes_ = 3;
        input_keys_ = new char* [3] {"feature_0_data", "feature_1_data", "feature_2_data"};
        input_shape_indptr_ = new mx_uint[4]{0, 2, 4, 6};
        input_shape_data_ = new mx_uint[6]{(mx_uint) batch_size, 20, (mx_uint) batch_size, 20, (mx_uint) batch_size, 12};
    }

    ~Model() {
        MXPredFree(out_);
    }

    // Predict service.
    void DoPredict(vector<vector<float> > &feature_data,
                   vector<string> &feature_name,
                   vector<int> &feature_sizes,
                   Matrix *result) {

        // Prepare input data.
        for (size_t i = 0; i < feature_name.size(); ++i) {
            const string &feat_name = feature_name[i];
            int size = feature_sizes[i] * batch_size_;
            MXPredSetInput(out_, feat_name.c_str(), feature_data[i].data(), size);
        }

        // Do predict.
        MXPredForward(out_);

        // Get predicted result.
        mx_uint output_index = 0;
        mx_uint *shape = 0;
        mx_uint shape_len;

        MXPredGetOutputShape(out_, output_index, &shape, &shape_len);
        size_t size = 1;
        for (mx_uint k = 0; k < shape_len; ++k) size *= shape[k];
        result->data_ptr_ = new float[size];
        result->row_ = batch_size_;
        result->col_ = (int) (size / batch_size_);

        MXPredGetOutput(out_, output_index, result->mutable_data(), size);
    }

    // Load network symbol file and parameter file.
    void Load(const string &symbol_file, const string &param_file) {
        symbol_data_ = new BufferFile(symbol_file);
        param_data_ = new BufferFile(param_file);
    }

public:
    void Init(TaskContext *context) {
        MXPredCreate((const char *) symbol_data_->GetBuffer(),
                     (const char *) param_data_->GetBuffer(),
                     static_cast<size_t>(param_data_->GetLength()),
                     dev_type_,
                     dev_id_,
                     num_input_nodes_,
                     (const char **) input_keys_,
                     input_shape_indptr_,
                     input_shape_data_,
                     &out_);
    }

private:
    int dev_type_ = 1; // 1: cpu, 2: gpu
    int dev_id_ = 0; // arbitrary
    mx_uint num_input_nodes_;

    mx_uint batch_size_;

    PredictorHandle out_ = 0;

    // Model symbols and params
    BufferFile *symbol_data_;
    BufferFile *param_data_;

    mx_uint *input_shape_indptr_;
    mx_uint *input_shape_data_;
    char **input_keys_;
};

