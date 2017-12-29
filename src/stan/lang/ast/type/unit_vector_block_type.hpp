#ifndef STAN_LANG_AST_UNIT_VECTOR_BLOCK_TYPE_HPP
#define STAN_LANG_AST_UNIT_VECTOR_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * Unit vector block var type.
     */
    struct unit_vector_block_type {
      /**
       * Unit vector length
       */
      expression K_;

      /**
       * Construct a block var type with default values.
       */
      unit_vector_block_type();

      /**
       * Construct a block var type with specified values.
       *
       * @param K size
       */
      unit_vector_block_type(const expression& K);
    };

  }
}
#endif
