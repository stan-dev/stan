#ifndef STAN_LANG_AST_VECTOR_BLOCK_TYPE_HPP
#define STAN_LANG_AST_VECTOR_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>

namespace stan {
  namespace lang {

    /**
     * Vector block var type.
     */
    struct vector_block_type {
      /**
       * Bounds constraints
       */
      range bounds_;

      /**
       * Vector length
       */
      expression N_;

      /**
       * Construct a block var type with default values.
       */
      vector_block_type();

      /**
       * Construct a block var type with specified values.
       * Length should be int expression - constructor doesn't check.
       *
       * @param bounds variable upper and/or lower bounds
       * @param N vector length
       */
      vector_block_type(const range& bounds,
                        const expression& N);

      /**
       * Get bounds.
       */
      range bounds() const;

      /**
       * Get N (num rows).
       */
      expression N() const;
    };
  }
}
#endif
