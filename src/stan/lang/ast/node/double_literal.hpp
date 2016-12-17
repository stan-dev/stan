#ifndef STAN_LANG_AST_NODE_DOUBLE_LITERAL_HPP
#define STAN_LANG_AST_NODE_DOUBLE_LITERAL_HPP

#include <stan/lang/ast/expr_type.hpp>

namespace stan {
  namespace lang {

    /**
     * Node for holding a double literal.
     */
    struct double_literal {
      /**
       * Value of literal.
       */
      double val_;

      /**
       * Expression type.
       */
      expr_type type_;

      /**
       * Default constructor for double literal.
       */
      double_literal();

      /**
       * Construct a double literal with the specified value.
       *
       * @param val value of literal
       */
      double_literal(double val);  // NOLINT(runtime/explicit)

      /**
       * Assign a double literal to this literal and return it.
       *
       * @param dl new value literal
       * @return new value reference
       */
      double_literal& operator=(const double_literal& dl);
    };


  }
}
#endif
