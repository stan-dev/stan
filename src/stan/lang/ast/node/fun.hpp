#ifndef STAN_LANG_AST_NODE_FUN_HPP
#define STAN_LANG_AST_NODE_FUN_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <string>
#include <vector>


namespace stan {
  namespace lang {

    struct expression;

    /**
     * Structure for function application.
     */
    struct fun {
      /**
       * Name of function being applied.
       */
      std::string name_;

      /**
       * Original name of function being applied (before name
       * transformation). 
       */
      std::string original_name_;

      /**
       * Sequence of argument expressions for function.
       */
      std::vector<expression> args_;

      /**
       * Type of result of applying function to arguments.
       */
      expr_type type_;

      /**
       * Construct a default function object.
       */
      fun();

      /**
       * Construct a function object with the specified name and
       * arguments. 
       *
       * @param name name of function
       * @param args sequence of arguments to function
       */
      fun(const std::string& name, const std::vector<expression>& args);
    };

  }
}
#endif
