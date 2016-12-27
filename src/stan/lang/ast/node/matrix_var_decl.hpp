#ifndef STAN_LANG_AST_NODE_MATRIX_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_MATRIX_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a matrix variable declaration.
     */
    struct matrix_var_decl : public base_var_decl {
      /**
       * Optional lower and upper bound constraints.
       */ 
      range range_;

      /**
       * Number of rows.
       */
      expression M_;

      /**
       * Number of columns.
       */
      expression N_;

      /**
       * Construct a matrix variable declaration with default values.
       */
      matrix_var_decl();

      /**
       * Construct a matrix variable declaration with the specified
       * range constraints, number of rows, number of columns, name,
       * number of array dimensions, and definition.
       *
       * @param range lower and upper bounds on values
       * @param M number of rows
       * @param N number of columns
       * @param name variable name
       * @param dims array dimension sizes
       * @param def defition of variable
       */
      matrix_var_decl(const range& range,
                      const expression& M,
                      const expression& N,
                      const std::string& name,
                      const std::vector<expression>& dims,
                      const expression& def);
    };
  }
}
#endif
