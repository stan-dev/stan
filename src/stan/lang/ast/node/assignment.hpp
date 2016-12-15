#ifndef STAN_LANG_AST_NODE_ASSIGNMENT_HPP
#define STAN_LANG_AST_NODE_ASSIGNMENT_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/variable_dims.hpp>

namespace stan {
  namespace lang {

    /**
     * AST node for assignment statements.
     */
    struct assignment {
      /**
       * Construct an empty assignment.
       */
      assignment();

      /**
       * Construct an assignment with an indexed variable and value. 
       *
       * @param var_dims variable and array/matrix indexes
       * @param expr right hand side, value being assigned to indexed variable
       */
      assignment(variable_dims& var_dims, expression& expr);

      /**
       * Variable plus indexes.  For example, <code>a[2,3]</code> for
       * variable <code>a</code> and indexes <code>2</code> and
       * <code>3</code> 
       */
      variable_dims var_dims_;

      /**
       * Value being assigned, which appears on the right hand side of
       * the assignment.
       */
      expression expr_;

      /**
       * Type of the left hand side variable before indexing.
       */
      base_var_decl var_type_;
    };

  }
}
#endif
