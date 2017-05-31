#ifndef STAN_LANG_AST_NODE_COMPOUND_ASSIGNMENT_HPP
#define STAN_LANG_AST_NODE_COMPOUND_ASSIGNMENT_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/variable_dims.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * AST node for compound assignment statements.
     */
    struct compound_assignment {

      /**
       * Type of the left hand side variable before indexing.
       */
      base_var_decl var_type_;

      /**
       * Variable plus indexes.  For example, <code>a[2,3]</code> for
       * variable <code>a</code> and indexes <code>2</code> and
       * <code>3</code> 
       */
      variable_dims var_dims_;

      /**
       * String representation of the arithmetic operation.
       * Doesn't include assignment operator <code>=<code>,
       * e.g. for compound addition-assignment <code>+=</code> stmt
       * <code>op</code> value is <code>+</code>.
       */
      std::string op_;

      /**
       * Value being assigned, which appears on the right hand side of
       * the compound assignment.
       */
      expression expr_;

      /**
       * Function signature of operation, is nil when lhs and rhs are scalars.
       */
      expression fun_sig_;

      /**
       * Construct an empty compound assignment.
       */
      compound_assignment();

      /**
       * Construct a compound assignment with an indexed variable, 
       * arithmetic operator, and value.
       *
       * @param var_dims variable and array/matrix indexes
       * @param op arithmetic operator
       * @param expr right hand side, value being assigned to indexed variable
       */
      compound_assignment(variable_dims& var_dims, std::string& op, expression& expr);
    };

  }
}
#endif
