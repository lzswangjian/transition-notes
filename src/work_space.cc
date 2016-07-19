#include "work_space.h"

string WorkspaceRegistry::DebugString() const {
  string str = "debug";
  return str;
}

VectorIntWorkspace::VectorIntWorkspace(int size) : elements_(size) {}

VectorIntWorkspace::VectorIntWorkspace(const vector<int> &elements)
        : elements_(elements) {}

VectorIntWorkspace::VectorIntWorkspace(int size, int value)
        : elements_(size, value) {}

string VectorIntWorkspace::TypeName() {
    return "Vector";
}

VectorVectorIntWorkspace::VectorVectorIntWorkspace(int size)
        : elements_(size) {}

string VectorVectorIntWorkspace::TypeName() {
    return "VectorVector";
}