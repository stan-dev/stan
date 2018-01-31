#ifndef STAN_LANG_AST_NODE_DOUBLE_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_DOUBLE_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    double_local_var_decl::double_local_var_decl() { }

    double_local_var_decl::double_local_var_decl(const std::string& name)
      : var_decl(name, double_type()) { }

    double_local_var_decl::double_local_var_decl(const std::string& name,
                                                 const expression& def)
      : var_decl(name, double_type(), def) { }
  }
}
#endif
