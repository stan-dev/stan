#ifndef STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    var_decl::var_decl() { }

    var_decl::var_decl(const std::string& name)
      : name_(name) {  }

    var_decl::var_decl(const std::string& name,
                       const bare_expr_type& type)
      : name_(name), bare_type_(type) { }

    var_decl::var_decl(const std::string& name,
                       const bare_expr_type& type,
                       const expression& def)
      : name_(name), bare_type_(type), def_(def) {  }
  }
}

#endif
