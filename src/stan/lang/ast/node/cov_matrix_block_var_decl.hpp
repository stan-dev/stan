#ifndef STAN_LANG_AST_NODE_COV_MATRIX_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_COV_MATRIX_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a covariance matrix variable declaration.
     */
    struct cov_matrix_block_var_decl : public var_decl {
      /**
       * Type object specifies number of rows and columns.
       */
      cov_matrix_block_type type_;

      /**
       * Construct a variable declaration for a covariance matrix.
       */
      cov_matrix_block_var_decl();

      /**
       * Construct a variable declaration for a covariance matrix
       * with the specified name and size.
       *
       * @param name variable name
       * @param K cov matrix size
       */
      cov_matrix_block_var_decl(const std::string& name,
                                const expression& K);

      /**
       * Construct a variable declaration for a covariance matrix
       * with the specified name, size, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param K cov matrix size
       * @param def defition of variable
       */
      cov_matrix_block_var_decl(const std::string& name,
                                const expression& K,
                                const expression& def);
    };
  }
}
#endif
