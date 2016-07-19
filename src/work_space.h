// Notes on thread-safety: All of the classes here are thread-compatible.
// More specifically, the registry machinery is thread-safe, as long as each
// thread performs feature extraction on a different Sentence object.

#ifndef WORK_SPACE_H_
#define WORK_SPACE_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <typeindex>
#include "base.h"

using namespace std;

/*!
 * \brief base class for shared workspaces. Derived classes implement a static member
 * function TypeName() which returns a human readable string name for the class.
 */
class Workspace {
  public:
    virtual ~Workspace() {}

  protected:
    Workspace() {}
};

/*!
 * \brief A registry that keeps track of workspaces.
 */
class WorkspaceRegistry {
public:
    WorkspaceRegistry() {}

    // Returns the index of a named workspace, adding it to the registry first
    // if necessary.
    template <class W>
    int Request(const string &name) {
        const std::type_index id = std::type_index(typeid(W));
        workspace_types_[id] = W::TypeName();
        vector<string> &names = workspace_names_[id];
        for (int i = 0; i < names.size(); ++i) {
            if (names[i] == name) return i;
        }
        names.push_back(name);
        return names.size() - 1;
    }

    const std::unordered_map<std::type_index, vector<string>> &WorkspaceNames() const {
        return workspace_names_;
    };

    string DebugString() const;

private:
    // Workspace type names, indexed as workspace_types_[typeid].
    std::unordered_map<std::type_index, string> workspace_types_;

    // Workspace names, indexed as workspace_names_[typeid][workspace].
    std::unordered_map<std::type_index, vector<string>> workspace_names_;

};

/*!
 * \brief A typed collected of workspaces. The workspaces are indexed according to an
 * external WorkspaceRegistry. If the WorkspaceSet is const, the contents are also
 * immutable.
 */
class WorkspaceSet {
public:
    ~WorkspaceSet() {}

    // Returns true if a workspace has been set.
    template <class W>
    bool Has(int index) const {
    }

    // Returns an indexed workspace; the workspace must have been set.
    template <class W>
    const W &Get(int index) const {
    }

    // Sets an indexed workspace; this takes ownership of the
    // workspace, which must have been new-allocated. It is an
    // error to set a workspace twice.
    template <class W>
    void Set(int index, W *workspace) {
    }

    void Reset(const WorkspaceRegistry &registry) {
    }

private:
    // The set of workspaces, indexed as workspaces_[typeid][index].
    std::unordered_map<std::type_index, vector<Workspace *> > workspaces_;
};


// A workspace that wraps around a single int.
class SingletonIntWorkspace : public Workspace {
  public:
    SingletonIntWorkspace() {}

    explicit SingletonIntWorkspace(int value) : value_(value) {}

    static string TypeName() { return "SingletonInt"; }

    int get() const { return value_; }

    void set(int value) { value_ = value; }

  private:
    // The enclosed int.
    int value_ = 0;
};

// A workspace that wraps around a vector of int.
class VectorIntWorkspace : public Workspace {
  public:
    explicit VectorIntWorkspace(int size);

    explicit VectorIntWorkspace(const vector<int> &elements);

    VectorIntWorkspace(int size, int value);

    static string TypeName();

    int element(int i) const { return elements_[i]; }

    void set_element(int i, int value) { elements_[i] = value; }

  private:
    vector<int> elements_;
};

class VectorVectorIntWorkspace : public Workspace {
  public:
    explicit VectorVectorIntWorkspace(int size);

    static string TypeName();

    const vector<int> &elements(int i) const {
      return elements_[i];
    }

    vector<int> *mutable_elements(int i) {
      return &(elements_[i]);
    }

  private:
    vector<vector<int> > elements_;
};

#endif
