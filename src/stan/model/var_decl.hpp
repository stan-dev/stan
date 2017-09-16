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
      const bool has_lower_bound_;
      const bool has_upper_bound_;

    public:
      /**
       * Construct a variable declaration with the specified name and
       * type name.
       *
       * @param[in] name name of variable
       * @param[in] type_name name of the variable's type
       * @param[in] array_dims number of array dimensions
       * @param[in] has_lower_bound true if declaration has lower
       * bound (default value false)
       * @param[in] has_upper_bound true if declaration has upper
       * bound (default value false)
       */
      var_decl(const std::string& name, const std::string& type_name,
               int array_dims, bool has_lower_bound = false,
               bool has_upper_bound = false)
        : name_(name), type_name_(type_name), array_dims_(array_dims),
          has_lower_bound_(has_lower_bound), has_upper_bound_(has_upper_bound)
      { }

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

      /**
       * Return true if this variable declaration includes a lower
       * bound constraint.
       *
       * @return true if declaration has lower bound constraint
       */
      bool has_lower_bound() const {
        return has_lower_bound_;
      }

      /**
       * Return true if this variable declaration includes an upper
       * bound constraint.
       *
       * @return true if declaration has upper bound constraint
       */
      bool has_upper_bound() const {
        return has_upper_bound_;
      }
    };

  }
}
#endif
