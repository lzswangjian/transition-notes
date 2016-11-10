#include "task_context.h"


TaskInput *TaskContext::GetInput(const string &name) {
    // Return existing input if it exists.
    for (int i = 0; i < spec_.input_size(); ++i) {
        if (spec_.input(i).name() == name) return spec_.mutable_input(i);
    }

    // Create new input.
    TaskInput *input = spec_.add_input();
    input->set_name(name);
    return input;
}

TaskInput *TaskContext::GetInput(const string &name, const string &file_format,
                                 const string &record_format) {
    return nullptr;
}

void TaskContext::SetParameter(const string &name, const string &value) {
    for (int i = 0; i < spec_.parameter_size(); ++i) {
        if (spec_.parameter(i).name() == name) {
            spec_.mutable_parameter(i)->set_value(value);
            return;
        }
    }

    // Add new parameter.
    TaskSpec::Parameter *param = spec_.add_parameter();
    param->set_name(name);
    param->set_value(value);
}

string TaskContext::GetParameter(const string &name) const {
    for (int i = 0; i < spec_.parameter_size(); ++i) {
        if (spec_.parameter(i).name() == name) return spec_.parameter(i).value();
    }
    // Parameter not found, return empty string.
    return "";
}

int TaskContext::GetIntParameter(const string &name) const {
    string value = GetParameter(name);
    return utils::ParseUsing<int>(value, 0, utils::ParseInt32);
}

int64_t TaskContext::GetInt64Parameter(const string &name) const {
    string value = GetParameter(name);
    return utils::ParseUsing<int64_t>(value, 0ll, utils::ParseInt64);
}

bool TaskContext::GetBoolParameter(const string &name) const {
    string value = GetParameter(name);
    return value == "true";
}

double TaskContext::GetFloatParameter(const string &name) const {
    string value = GetParameter(name);
    return utils::ParseUsing<double>(value, 0.0, utils::ParseDouble);
}

string TaskContext::Get(const string &name, const string &defval) const {
    return Get(name, defval.c_str());
}

string TaskContext::Get(const string &name, const char *defval) const {
    for (int i = 0; i < spec_.parameter_size(); ++i) {
        if (spec_.parameter(i).name() == name) return spec_.parameter(i).value();
    }
    // Parameter not found, return empty string.
    return defval;
}

bool TaskContext::Get(const string &name, bool defval) const {
    string value = Get(name, "");
    return value.empty() ? defval : value == "true";
}

int TaskContext::Get(const string &name, int defval) const {
    string value = Get(name, "");
    return utils::ParseUsing<int>(value, defval, utils::ParseInt32);
}

int64_t TaskContext::Get(const string &name, int64_t defval) const {
    string value = Get(name, "");
    return utils::ParseUsing<int64_t>(value, defval, utils::ParseInt64);
}

double TaskContext::Get(const string &name, double defval) const {
    string value = Get(name, "");
    return utils::ParseUsing<double>(value, defval, utils::ParseDouble);
}

string TaskContext::InputFile(const TaskInput &input) {
    return input.part(0).file_pattern();
}

bool TaskContext::Supports(const TaskInput &input, const string &file_format,
                           const string &record_format) {
    return false;
}

void TaskContext::SetMode(bool is_train) {
    train = is_train;
}

bool TaskContext::GetMode() const {
    return train;
}















