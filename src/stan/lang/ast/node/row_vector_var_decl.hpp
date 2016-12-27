#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a row vector variable declaration.
     */
    struct row_vector_var_decl : public base_var_decl {
      /**
       * Option lower and upper bounds for values in the vector.
       */
      range range_;

      /**
       * Number of columns in the row vector (its size).
       */
      expression N_;

      /**
       * Construct a row vector variable declaration with default
       * values.
       */
      row_vector_var_decl();

      /**
       * Construct a row vector with the specified range constraint
       * (which has optional lower and upper bounds), number of
       * columns (size), name, array dimensions, and definition (which
       * may be nil to indicate it is not initialized) .
       *
       * @param range optional upper and lower bound on values
       * @param N number of columns (size)
       * @param name variable name
       * @param dims number of array dimensions
       * @param def definition
       */
      row_vector_var_decl(const range& range,
                          const expression& N,
                          const std::string& name,
                          const std::vector<expression>& dims,
                          const expression& def);
    };
  }
}
#endif
