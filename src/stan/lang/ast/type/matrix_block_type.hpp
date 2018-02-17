#ifndef STAN_LANG_AST_MATRIX_BLOCK_TYPE_HPP
#define STAN_LANG_AST_MATRIX_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>

namespace stan {
  namespace lang {

    /**
     * Matrix block var type.
     */
    struct matrix_block_type {
      /**
       * Bounds constraints
       */
      range bounds_;

      /**
       * Number of rows (arg_1)
       */
      expression M_;

      /**
       * Number of columns (arg_2)
       */
      expression N_;

      /**
       * Construct a block var type with default values.
       */
      matrix_block_type();

      /**
       * Construct a block var type with specified values.
       * Sizes should be int expressions - constructor doesn't check.
       *
       * @param bounds variable upper and/or lower bounds
       * @param M num rows
       * @param N num columns
       */
      matrix_block_type(const range& bounds,
                        const expression& M,
                        const expression& N);

      /**
       * Get bounds.
       */
      range bounds() const;

      /**
       * Get M (num rows).
       */
      expression M() const;

      /**
       * Get N (num cols).
       */
      expression N() const;
    };
  }
}
#endif
