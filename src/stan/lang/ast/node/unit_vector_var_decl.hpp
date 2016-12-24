#ifndef STAN_LANG_AST_NODE_UNIT_VECTOR_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_UNIT_VECTOR_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a unit vector. 
     */
    struct unit_vector_var_decl : public base_var_decl {
      /**
       * Unit vector size.
       */
      expression K_;

      /**
       * Construct a default unit vector declaration.
       */
      unit_vector_var_decl();

      /**
       * Construct a unit vector declaration with the specified size,
       * name, and number of array dimensions.
       *
       * @param K size of unit vector
       * @param name variable name
       * @param dims array dimension sizes
       * @param def definition
       */
      unit_vector_var_decl(const expression& K,
                           const std::string& name,
                           const std::vector<expression>& dims,
                           const expression& def);
    };
  }
}
#endif
