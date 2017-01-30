#ifndef STAN_LANG_AST_NODE_VARIABLE_DEF_HPP
#define STAN_LANG_AST_NODE_VARIABLE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    variable::variable() { }

    variable::variable(const std::string& name) : name_(name) { }

    void variable::set_type(const base_expr_type& base_type, size_t num_dims) {
      type_ = expr_type(base_type, num_dims);
    }

  }
}
#endif
