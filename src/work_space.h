#ifndef WORK_SPACE_H_
#define WORK_SPACE_H_

/*!
 * \brief base class for shared workspaces.
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

};

/*!
 * \brief A typed collected of workspaces.
 */
class WorkspaceSet {
};
#endif
