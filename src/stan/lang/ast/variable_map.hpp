#ifndef STAN_LANG_AST_VARIABLE_MAP_HPP
#define STAN_LANG_AST_VARIABLE_MAP_HPP

#include <stan/lang/ast/var_origin.hpp>
#include <stan/lang/ast/node/base_var_decl.hpp>
#include <cstddef>
#include <map>
#include <string>
#include <utility>

namespace stan {
  namespace lang {

    /**
     * A map from function names to their base declarations and their
     * origin.  
     */
    struct variable_map {
      /**
       * A variable type and the origin of its declaration.
       */
      typedef std::pair<base_var_decl, var_origin> range_t;

      /**
       * Return true if a variable has been declared with the
       * specified name. 
       *
       * @param name variable name
       * @return true if the variable has been declared
       */
      bool exists(const std::string& name) const;

      /**
       * Return the type for the variable with the specified name. 
       *
       * @param name variable name
       * @return base declaration for variable with the specified name 
       * @throw std::invalid_argument if the variable has not been
       * declared 
       */
      base_var_decl get(const std::string& name) const;

      /**
       * Return the type declared for the variable with the specified
       * name. 
       *
       * @param name variable name
       * @return base type
       * @throw std::invalid_argument if the variable has not been
       * declared 
       */
      base_expr_type get_base_type(const std::string& name) const;

      /**
       * Return the number of dimensions declared for the variable
       * with the specified name.
       *
       * @param name variable name
       * @return number of dimensions declared for variable
       * @throw std::invalid_argument if the variable has not been
       * declared 
       */
      std::size_t get_num_dims(const std::string& name) const;

      /**
       * Return the origin of the variable declaration for the
       * variable with the specified name.
       *
       * @param name variable name
       * @return origin for the variable
       * @throw std::invalid_argument if the variable has not been
       * declared 
       */
      var_origin get_origin(const std::string& name) const;

      /**
       * Add the specified declaration for a variable with the
       * specified name originating in the specified block.  Will
       * destructively overwrite the declaration of an existing
       * variable if called with a new declaration and origin. 
       *
       * @param name variable name
       * @param base_decl variable declaration
       * @param vo origin
       */
      void add(const std::string& name,
               const base_var_decl& base_decl,
               const var_origin& vo);

      /**
       * Remove the declaraiton for the variable with the specified
       * name.  If the variable had not already been declared, it
       * the function exits silently.
       *
       * @param name name of variable to remove
       */
      void remove(const std::string& name);

      /**
       * The stored map from function names to their declarations and
       * origins.
       */
      std::map<std::string, range_t> map_;
    };

  }
}
#endif
