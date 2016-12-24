#ifndef STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a simplex.
     */
    struct simplex_var_decl : public base_var_decl {
      /**
       * Simplex size.
       */
      expression K_;

      /**
       * Construct a default simplex declaration.
       */
      simplex_var_decl();

      /**
       * Construct a simplex declaration with the specified size,
       * name, and number of array dimensions.
       *
       * @param K size of simplex
       * @param name variable name
       * @param dims array dimension sizes
       * @param def definition
       */
      simplex_var_decl(const expression& K,
                       const std::string& name,
                       const std::vector<expression>& dims,
                       const expression& def);
    };
  }
}
#endif
