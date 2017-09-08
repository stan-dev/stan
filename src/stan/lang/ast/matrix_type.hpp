#ifndef STAN_LANG_AST_MATRIX_TYPE_HPP
#define STAN_LANG_AST_MATRIX_TYPE_HPP

namespace stan {
  namespace lang {

    /**
     * Matrix base expression type.
     */
    struct matrix_type {
      static const int ORDER_ID = 6;

      /**
       * Fixed numerical ID used for sorting.
       */
      int order_id_;
    };

  }
}
#endif
