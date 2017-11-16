#ifndef STAN_LANG_AST_NODE_FOREACH_M_STATEMENT_HPP
#define STAN_LANG_AST_NODE_FOREACH_M_STATEMENT_HPP

#include <stan/lang/ast/node/expression.hpp>
//#include <stan/lang/ast/node/array_expr.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * AST node for representing a foreach statement over a matrix.
     */
    struct foreach_m_statement {
      /**
       * Construct an uninitialized foreach statement.
       */
      foreach_m_statement();

      /**
       * Construct a foreach statement that loops the specified variable
       * over the specified expression to execute the specified statement.
       *
       * @param[in] variable loop variable
       * @param[in] expression value expression foreach loop variable
       * @param[in] stmt body of the foreach loop
       */
      foreach_m_statement(const std::string& variable,
                          const expression& expression, //FOREACHCHANGE: really, this should be an array_expr. where do we do the type checking?
                          const statement& stmt);

      /**
       * The loop variable.
       */
      std::string variable_;

      /**
       * The expression of values for the loop variable.
       */
      expression expression_; //FOREACHCHANGE:  really, this should be an array_expr. where do we do the type checking?

      /**
       * The body of the foreach loop.
       */
      statement statement_;
    };

  }
}
#endif
