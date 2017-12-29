#ifndef STAN_LANG_AST_ROW_VECTOR_LOCAL_TYPE_HPP
#define STAN_LANG_AST_ROW_VECTOR_LOCAL_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * Row vector local var type.
     */
    struct row_vector_local_type {
      /**
       * Row vector length
       */
      expression N_;

      /**
       * Construct a local var type with default values.
       */
      row_vector_local_type();

      /**
       * Construct a local var type with specified values.
       *
       * @param N vector length
       */
      row_vector_local_type(const expression& N);
    };

  }
}
#endif
