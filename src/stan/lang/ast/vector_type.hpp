#ifndef STAN_LANG_AST_VECTOR_TYPE_HPP
#define STAN_LANG_AST_VECTOR_TYPE_HPP

namespace stan {
  namespace lang {

    /**
     * Vector base expression type.
     */
    struct vector_type {
      static const int ORDER_ID = 4;

      /**
       * Fixed numerical ID used for sorting.
       */
      int order_id_;
    };

  }
}
#endif
