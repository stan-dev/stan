#ifndef STAN_LANG_AST_COV_MATRIX_BLOCK_TYPE_HPP
#define STAN_LANG_AST_COV_MATRIX_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * Covariance matrix block var type.
     */
    struct cov_matrix_block_type {
      /**
       * Number of rows and columns
       */
      expression K_;

      /**
       * Construct a block var type with default values.
       */
      cov_matrix_block_type();

      /**
       * Construct a block var type with specified values.
       *
       * @param K cov matrix size
       */
      cov_matrix_block_type(const expression& K);
    };

  }
}
#endif
