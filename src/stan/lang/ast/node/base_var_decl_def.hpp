#ifndef STAN_LANG_AST_NODE_BASE_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_BASE_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    base_var_decl::base_var_decl() { }

    base_var_decl::base_var_decl(const base_expr_type& base_type)
      : base_type_(base_type) {  }

    base_var_decl::base_var_decl(const std::string& name,
                                 const std::vector<expression>& dims,
                                 const base_expr_type& base_type)
      : name_(name), dims_(dims), base_type_(base_type) {  }

    base_var_decl::base_var_decl(const std::string& name,
                                 const std::vector<expression>& dims,
                                 const base_expr_type& base_type,
                                 const expression& def)
      : name_(name), dims_(dims), base_type_(base_type), def_(def) {  }

  }
}
#endif
