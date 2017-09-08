#ifndef STAN_LANG_AST_ILL_FORMED_TYPE_HPP
#define STAN_LANG_AST_ILL_FORMED_TYPE_HPP

namespace stan {
  namespace lang {

    /**
     * Ill_Formed base expression type.
     */
    struct ill_formed_type {
      static const int ORDER_ID = 0;

      /**
       * Fixed numerical ID used for sorting.
       */
      int order_id_;
    };

  }
}
#endif

