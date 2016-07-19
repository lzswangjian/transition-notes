//
// Created by ShengLi on 7/13/16.
//

#include "task_spec.h"


TaskInput *TaskSpec::add_input() {
    TaskInput *ti = new TaskInput(); // who release it ?
    input_.push_back(ti);
    return ti;
}

TaskSpec::Parameter *TaskSpec::add_parameter() {
    TaskSpec::Parameter *param = new TaskSpec::Parameter();
    parameter_.push_back(param);
    return param;
}



