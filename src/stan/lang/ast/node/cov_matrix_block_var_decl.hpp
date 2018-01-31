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
       * with the specified name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      cov_matrix_block_var_decl(const std::string& name,
                                const cov_matrix_block_type& type);

      /**
       * Construct a variable declaration for a covariance matrix
       * with the specified name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      cov_matrix_block_var_decl(const std::string& name,
                                const cov_matrix_block_type& type,
                                const expression& def);
    };
  }
}
#endif
