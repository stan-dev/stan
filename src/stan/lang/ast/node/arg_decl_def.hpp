#ifndef STAN_LANG_AST_NODE_ARG_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ARG_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    arg_decl::arg_decl() { }

    arg_decl::arg_decl(const expr_type& arg_type, const std::string& name)
      : arg_type_(arg_type), name_(name) { }

    base_var_decl arg_decl::base_variable_declaration() const {
      std::vector<expression> dims;
      for (size_t i = 0; i < arg_type_.num_dims_; ++i)
        dims.push_back(expression(int_literal(0)));  // dummy value 0
      return base_var_decl(name_, dims, arg_type_.base_type_);
    }

  }
}
#endif
