#ifndef STAN_LANG_AST_NODE_COV_MATRIX_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_COV_MATRIX_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a covariance matrix variable declaration.
     */
    struct cov_matrix_var_decl : public base_var_decl {
      /**
       * Number of rows and columns.
       */
      expression K_;

      /**
       * Construct a variable declaration for a covariance matrix.
       */
      cov_matrix_var_decl();

      /**
       * Construct a covariance matrix variable declaration for a
       * correlation matrix with the specified number of rows and
       * columns, name, and number of array dimensions.
       *
       * @param K number of rows and columns
       * @param name variable name
       * @param dims array dimension sizes
       * @param def defition of variable
       */
      cov_matrix_var_decl(const expression& K,
                          const std::string& name,
                          const std::vector<expression>& dims,
                          const expression& def);
    };
  }
}
#endif
