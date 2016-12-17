#ifndef STAN_LANG_AST_NODE_ARRAY_EXPR_HPP
#define STAN_LANG_AST_NODE_ARRAY_EXPR_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/var_origin.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <vector>

namespace stan {
  namespace lang {

    struct expresssion;

    /**
     * Structure to hold an array expression.
     */
    struct array_expr {
      /**
       * Sequence of expressions for array values.
       */
      std::vector<expression> args_;

      /**
       * Type of array.
       */
      expr_type type_;

      /**
       * True if there is a variable within any of the expressions
       * that is a parameter, transformed parameter, or non-integer
       * local variable.
       */
      bool has_var_;

      /**
       * Origin of this array expression.
       *
       */
      // TODO(carpenter): rename to "array_expr_origin_"
      var_origin var_origin_;

      /**
       * Construct a default array expression.
       */
      array_expr();

      /**
       * Construct an array expression from the specified sequence of
       * expressions.
       *
       * @param args sequence of arguments
       */
      array_expr(const std::vector<expression>& args);  // NOLINT

      /**
       * Assign specified array expression to this array expression.
       *
       * @param al new array expression value
       * @return reference to value
       */
      array_expr& operator=(const array_expr& al);
    };

  }
}
#endif
