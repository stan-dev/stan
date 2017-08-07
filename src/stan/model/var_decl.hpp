#ifndef STAN_MODEL_VAR_DECL_HPP
#define STAN_MODEL_VAR_DECL_HPP

#include <string>

namespace stan {
  namespace model {

    /**
     * Static variable declaration information including the name of
     * the variable, the name of its type, and the number of array
     * dimensions.
     */
    class var_decl {
    private:
      const std::string name_;
      const std::string type_name_;
      const int array_dims_;

    public:
      /**
       * Construct a variable declaration with the specified name and
       * type name.
       *
       * @param[in] name name of variable
       * @param[in] type_name name of the variable's type
       * @param[in] array_dims number of array dimensions
       */
      var_decl(const std::string& name, const std::string& type_name,
               int array_dims)
        : name_(name), type_name_(type_name), array_dims_(array_dims) { }

      /**
       * Return the name of the variable in this declaration.  The
       * lifespan of the reference is the same as for this class.
       *
       * @return name of variable declared
       */
      const std::string& name() const {
        return name_;
      }

      /**
       * Return the name of the type of the variable in this
       * declaration.  The lifespan of the reference is the same as
       * for this class.
       *
       * @return name of the type declared
       */
      const std::string& type_name() const {
        return type_name_;
      }

      /**
       * Return the number of array dimensions in this declarations.
       *
       * @return number of array dimensions
       */
      int array_dims() const {
        return array_dims_;
      }
    };

  }
}
#endif
