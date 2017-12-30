#ifndef STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_HPP
#define STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * Cholesky factor matrix block var type.
     */
    struct cholesky_factor_block_type {
      /**
       * Number of rows.
       */
      expression M_;

      /**
       * Number of columns.
       */
      expression N_;

      /**
       * Construct a block var type with default values.
       */
      cholesky_factor_block_type();

      /**
       * Construct a block var type with specified values.
       * Sizes should be int expressions - constructor doesn't check.
       *
       * @param M num rows
       * @param N num columns
       */
      cholesky_factor_block_type(const expression& M,
                                 const expression& N);
    };

  }
}
#endif
