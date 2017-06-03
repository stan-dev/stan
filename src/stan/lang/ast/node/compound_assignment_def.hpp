#ifndef STAN_LANG_AST_NODE_COMPOUND_ASSIGNMENT_DEF_HPP
#define STAN_LANG_AST_NODE_COMPOUND_ASSIGNMENT_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    compound_assignment::compound_assignment() { }

    compound_assignment::compound_assignment(variable_dims& var_dims,
                                             std::string& op,
                                             expression& expr)
      : var_dims_(var_dims), op_(op), expr_(expr) { }

  }
}
#endif
