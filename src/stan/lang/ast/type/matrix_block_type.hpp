#ifndef STAN_LANG_AST_MATRIX_BLOCK_TYPE_HPP
#define STAN_LANG_AST_MATRIX_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/node/locscale.hpp>

namespace stan {
  namespace lang {
  // TODO(VMatthijs): We should only allow to have either a range or a locscale.

    /**
     * Matrix block var type.
     */
    struct matrix_block_type {
      /**
       * Bounds constraints
       */
      range bounds_;

      /**
       * Location and scale
       */
      locscale ls_;

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
       * @param ls variable location and scale
       * @param M num rows
       * @param N num columns
       */
      matrix_block_type(const range& bounds,
                        const locscale& ls,
                        const expression& M,
                        const expression& N);

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
       * Construct a block var type with specified values.
       * Sizes should be int expressions - constructor doesn't check.
       *
       * @param ls variable location and scale
       * @param M num rows
       * @param N num columns
       */
      matrix_block_type(const locscale& ls,
                        const expression& M,
                        const expression& N);

      /**
       * Get bounds.
       */
      range bounds() const;

      /**
       * Get location and scale.
       */
      locscale ls() const;

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
