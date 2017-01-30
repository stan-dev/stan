#ifndef STAN_LANG_AST_NODE_VECTOR_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_VECTOR_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a column vector variable declaration.
     */
    struct vector_var_decl : public base_var_decl {
      /**
       * Option lower and upper bounds for values in the vector.
       */
      range range_;

      /**
       * Number of rows in the column vector (its size).
       */
      expression M_;

      /**
       * Construct a column vector variable declaration with default
       * values.
       */
      vector_var_decl();

      /**
       * Construct a column vector with the specified range constraint
       * (which has optional lower and upper bounds), number of rows,
       * name, array dimensions, and definition (which may be nil to
       * indicate it is not initialized) .
       *
       * @param range optional upper and lower bound on values
       * @param M number of rows (size)
       * @param name variable name
       * @param dims number of array dimensions
       * @param def definition
       */
      vector_var_decl(const range& range,
                      const expression& M,
                      const std::string& name,
                      const std::vector<expression>& dims,
                      const expression& def);
    };
  }
}
#endif
