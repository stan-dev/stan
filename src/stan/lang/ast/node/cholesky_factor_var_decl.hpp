#ifndef STAN_LANG_AST_NODE_CHOLESKY_FACTOR_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_CHOLESKY_FACTOR_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a Cholesky factor variable declaration.
     */
    struct cholesky_factor_var_decl : public base_var_decl {
      /**
       * Number of rows.
       */
      expression M_;

      /**
       * Number of columns.
       */
      expression N_;

      /**
       * Construct a Cholesky factor variable declaration with default
       * values.
       */
      cholesky_factor_var_decl();

      /**
       * Construct a Cholesky factor variable declaration with the
       * specified number of rows, number of columns, name, and number
       * of array dimensions
       *
       * @param M number of rows
       * @param N number of columns
       * @param name variable name
       * @param dims array dimension sizes
       * @param def defition of variable
       */
      cholesky_factor_var_decl(const expression& M,
                               const expression& N,
                               const std::string& name,
                               const std::vector<expression>& dims,
                               const expression& def);
    };
  }
}
#endif
