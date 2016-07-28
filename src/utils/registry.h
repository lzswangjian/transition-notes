#ifndef REGISTRY_H
#define REGISTRY_H

/*!
 * \brief Registry for component registration.
 *
 * These classes can be used for creating registries of components
 * conforming to the same interface.
 * This is useful for making a component-based architecture where
 * the specific implementation classes can be selected at runtime.
 * There is supprot for both class-based and instance based registries.
 *
 */
#include <string.h>
#include <string>
#include <vector>

#include "utils.h"

// Component metadata with information about name, class, and 
// code location.
class ComponentMetadata {
public:
    ComponentMetadata(const char *name, const char *class_name,
                      const char *file, int line)
            : name_(name),
              class_name_(class_name),
              file_(file),
              line_(line),
              link_(NULL) {}

    // Returns component name.
    const char *name() const { return name_; }

    // Metadata objects can be linked in a list.
    ComponentMetadata *link() const { return link_; }

    void set_link(ComponentMetadata *link) { link_ = link; }

private:
    // Component name.
    const char *name_;

    // Name of class for component.
    const char *class_name_;

    // Code file and location where the component was registered.
    const char *file_;
    int line_;

    // Link to next metadata object in list.
    ComponentMetadata *link_;
};

/*!
 * \brief The master registry contains all registered component registries.
 * A registry is not registered in the master registry until the first component
 * of that type is registered.
 */
class RegistryMetadata : public ComponentMetadata {
public:
    RegistryMetadata(const char *name, const char *class_name, const char *file,
                     int line, ComponentMetadata **components)
            : ComponentMetadata(name, class_name, file, line), components_(components) {}

    // Registers a component registry in the master registry.
    static void Register(RegistryMetadata *registry);

private:
    // Location of list of components in registry.
    ComponentMetadata **components_;
};

/*!
 * \brief Registry for components.
 * An object can be registered with a type name in the registry. The named instances
 * in the registry can be returned using the Lookup() method. The components in the
 * registry are put into a linked list of components. It is important that the component
 * registry can be statically initialized in order not to depend on initialization order.
 */
template<class T>
struct ComponentRegistry {
    typedef ComponentRegistry<T> Self;

    // Component registration class.
    class Registrar : public ComponentMetadata {
    public:
        Registrar(Self *registry, const char *type, const char *class_name,
                  const char *file, int line, T *object)
                : ComponentMetadata(type, class_name, file, line), object_(object) {
            if (registry->components == NULL) {
                RegistryMetadata::Register(new RegistryMetadata(
                        registry->name, registry->class_name, registry->file,
                        registry->line,
                        reinterpret_cast<ComponentMetadata **>(&registry->components)));
            }

            set_link(registry->components);
            registry->components = this;
        }

        const char *type() const { return name(); }

        T *object() const { return object_; }

        Registrar *next() const { return static_cast<Registrar *>(link()); }

    private:
        // Component object.
        T *object_;
    };

    // Finds registrar for named component in registry.
    const Registrar *GetComponent(const char *type) const {
        Registrar *r = components;
        while (r != NULL && strcmp(type, r->type()) != 0) r = r->next();
        if (r == NULL) {
            LOG(FATAL) << "Unknown " << name << " component: '" << type << "'.";
        }
        return r;
    }

    T *Lookup(const char *type) const { return GetComponent(type)->object(); }

    T *Lookup(const string &type) const { return Lookup(type.c_str()); }

    // Textual description of the kind of components in the registry.
    const char *name;

    // Base class name of component type.
    const char *class_name;

    // File and line where the registry is defined.
    const char *file;
    int line;

    // Linked list of registered components.
    Registrar *components;
};


// Base class for registerable class-based components.
template<class T>
class RegisterableClass {
public:
    typedef T *(Factory)();

    typedef ComponentRegistry<Factory> Registry;

    static T *Create(const string &type) { return registry()->Lookup(type)(); }

    static Registry *registry() { return &registry_; }

private:
    // Registry for class.
    static Registry registry_;
};

// Base class for registerable instance-based components.
template<class T>
class RegisterableIntance {
public:
    // Registry type.
    typedef ComponentRegistry<T> Registry;

private:
    static Registry registry_;
};

#define REGISTER_CLASS_COMPONENT(base, type, component) \
  static base *__##component##__factory() { return new component; } \
  static base::Registry::Registrar __##component##__##registrar( \
      base::registry(), type, #component, __FILE__, __LINE__, \
      __##component##__factory)

#define REGISTER_CLASS_REGISTRY(type, classname) \
  template <> \
  classname::Registry RegisterableClass<classname>::registry_ = { \
    type, #classname, __FILE__, __LINE__, NULL}

#define REGISTER_INSTANCE_COMPONENT(base, type, component) \
  static base::Registry::Registrar __##component##__##registrar( \
      base::registry(), type, #component, __FILE__, __LINE__, new component)

#define REGISTER_INSTANCE_REGISTRY(type, classname) \
  template <> \
  classname::Registry RegisterableIntance<classname>::registry_ = { \
    type, #classname, __FILE__, __LINE__, NULL}

#endif /* REGISTRY_H */