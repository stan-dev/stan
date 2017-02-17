#ifndef STAN_LANG_AST_NODE_VECTOR_EXPR_HPP
#define STAN_LANG_AST_NODE_VECTOR_EXPR_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/scope.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <vector>

namespace stan {
  namespace lang {

    struct expresssion;

    /**
     * Structure to hold a column vector expression.
     */
    struct vector_expr {
      /**
       * Sequence of expressions for vector values.
       */
      std::vector<expression> args_;

      /**
       * Number of rows in the column vector (its size).
       */
      expression N_;

      /**
       * True if there is a variable within any of the expressions
       * that is a parameter, transformed parameter, or non-integer
       * local variable.
       */
      bool has_var_;

      /**
       * Scope of this vector expression.
       *
       */
      scope vector_expr_scope_;

      /**
       * Construct a default vector expression.
       */
      vector_expr();

      /**
       * Construct an vector expression from the specified sequence of
       * expressions.
       *
       * @param args sequence of arguments
       */
      vector_expr(const std::vector<expression>& args, const expression& N);

      /**
       * Assign specified vector expression to this vector expression.
       *
       * @param al new vector expression value
       * @return reference to value
       */
      vector_expr& operator=(const vector_expr& al);
    };

  }
}
#endif
