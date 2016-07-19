#ifndef SYNTAXNET_TASK_SPEC_H
#define SYNTAXNET_TASK_SPEC_H

#include <string>
#include <vector>

using namespace std;

class TaskInput {
public:
    string name_;
    const string &name() const { return name_; }
    void set_name(const string &name) { name_ = name; }

    string creator_;

    string file_format_;

    // Record format for resource.
    string record_format_;

    bool multi_file_;

    struct Part {
        string file_pattern_;
        const string &file_pattern() const { return file_pattern_; }

        string file_format_;

        string record_format_;
    };

    vector<Part *> part_;

    const Part &part(int index) const {
        return *part_[index];
    }
};

class TaskOutput {
public:
    string name_;

    string file_format_;

    string record_format_;

    int32_t shards_;

    string file_base_;

    string file_extension_;
};

class TaskSpec {
public:
    string task_name_;

    // Workflow task type.
    string task_type_;

    // Task inputs.
    vector<TaskInput *> input_;

    // Task outputs.
    vector<TaskOutput *> output_;

    struct Parameter {
        string name_;
        const string &name() const { return name_; }
        void set_name(const string &name) { name_ = name; }

        string value_;
        const string &value() const { return value_; }
        void set_value(const string &value) { value_ = value; }
    };

    vector<Parameter *> parameter_;
    int parameter_size() const { return parameter_.size(); }
    const Parameter &parameter(int index) const {
        return *parameter_[index];
    }
    Parameter *mutable_parameter(int index) {
        return parameter_[index];
    }
    Parameter *add_parameter();

    // access functions.

    int input_size() const { return input_.size(); }

    const TaskInput &input(int index) const {
        return *input_[index];
    }

    TaskInput *mutable_input(int index) {
        return input_[index];
    }

    TaskInput *add_input();
};

#endif //SYNTAXNET_TASK_SPEC_H
