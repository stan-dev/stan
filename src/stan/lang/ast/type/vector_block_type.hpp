#ifndef STAN_LANG_AST_VECTOR_BLOCK_TYPE_HPP
#define STAN_LANG_AST_VECTOR_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/node/locscale.hpp>

namespace stan {
  namespace lang {
  // TODO(VMatthijs): We should only allow to have either a range or a locscale.

    /**
     * Vector block var type.
     */
    struct vector_block_type {
      /**
       * Bounds constraints
       */
      range bounds_;

      /**
       * Location and scale
       */
      locscale ls_;

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
       * @param ls variable location and scale
       * @param N vector length
       */
      vector_block_type(const range& bounds, const locscale& ls,
                        const expression& N);

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
       * Construct a block var type with specified values.
       * Length should be int expression - constructor doesn't check.
       *
       * @param ls variable location and scale
       * @param N vector length
       */
      vector_block_type(const locscale& ls,
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
       * Get N (num rows).
       */
      expression N() const;
    };
  }
}
#endif
