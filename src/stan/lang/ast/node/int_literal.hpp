#ifndef STAN_LANG_AST_NODE_INT_LITERAL_HPP
#define STAN_LANG_AST_NODE_INT_LITERAL_HPP

#include <stan/lang/ast/expr_type.hpp>

namespace stan {
  namespace lang {

    struct int_literal {
      /**
       * Value of literal.
       */
      int val_;

      /**
       * Expression type of literal.
       */
      expr_type type_;

      /**
       * Construct a default int literal.
       */
      int_literal();

      /**
       * Construct an int literal with the specified value.
       *
       * @param val value of literal
       */
      int_literal(int val);  // NOLINT(runtime/explicit)

      /**
       * Copy constructor.
       *
       * @param il value
       */
      int_literal(const int_literal& il);  // NOLINT(runtime/explicit)

      /**
       * Assignment for int literals.
       *
       * @param il new value
       * @return reference to literal assigned
       */
      int_literal& operator=(const int_literal& il);
    };
  }
}
#endif
