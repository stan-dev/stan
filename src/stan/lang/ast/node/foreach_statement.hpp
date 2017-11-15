#ifndef STAN_LANG_AST_NODE_FOREACH_STATEMENT_HPP
#define STAN_LANG_AST_NODE_FOREACH_STATEMENT_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * AST node for representing a foreach statement.
     */
    struct foreach_statement {
      /**
       * Construct an uninitialized foreach statement.
       */
      foreach_statement();

      /**
       * Construct a foreach statement that loops the specified variable
       * over the specified range to execute the specified statement.
       *
       * @param[in] variable loop variable
       * @param[in] range value range foreach loop variable
       * @param[in] stmt body of the foreach loop
       */
      foreach_statement(const std::string& variable, const range& range, //FOREACHCHANGE: this needs to be changed (range)
                        const statement& stmt);

      /**
       * The loop variable.
       */
      std::string variable_;

      /**
       * The range of values for the loop variable.
       */
      range range_; //FOREACHCHANGE: this needs to be changed (range)

      /**
       * The body of the foreach loop.
       */
      statement statement_;
    };

  }
}
#endif
