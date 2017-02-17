#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/scope.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <vector>

namespace stan {
  namespace lang {

    struct expresssion;

    /**
     * Structure to hold a row vector expression.
     */
    struct row_vector_expr {
      /**
       * Sequence of expressions for row vector values.
       */
      std::vector<expression> args_;

      /**
       * Expression type.
       */
      expr_type type_;

      /**
       * True if there is a variable within any of the expressions
       * that is a parameter, transformed parameter, or non-integer
       * local variable.
       */
      bool has_var_;

      /**
       * Scope of this row vector expression.
       *
       */
      scope row_vector_expr_scope_;

      /**
       * Construct a default row vector expression.
       */
      row_vector_expr();

      /**
       * Construct an row vector expression from the specified sequence of
       * expressions.
       *
       * @param args sequence of arguments
       */
      row_vector_expr(const std::vector<expression>& args);  // NOLINT

      /**
       * Assign specified row vector expression to this row vector expression.
       *
       * @param al new row vector expression value
       * @return reference to value
       */
      row_vector_expr& operator=(const row_vector_expr& al);
    };

  }
}
#endif
