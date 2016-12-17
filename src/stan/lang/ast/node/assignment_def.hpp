#ifndef STAN_LANG_AST_NODE_ASSIGNMENT_DEF_HPP
#define STAN_LANG_AST_NODE_ASSIGNMENT_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    assignment::assignment() { }

    assignment::assignment(variable_dims& var_dims, expression& expr)
      : var_dims_(var_dims), expr_(expr) { }

  }
}
#endif
