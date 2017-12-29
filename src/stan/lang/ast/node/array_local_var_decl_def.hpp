#ifndef STAN_LANG_AST_NODE_ARRAY_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ARRAY_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    array_local_var_decl::array_local_var_decl() { }

    array_local_var_decl::array_local_var_decl(
                          const std::string& name,
                          const local_var_type& el_type,
                          const expression& len,
                          const expression& def)
      : var_decl(name, array_bare_type(), def),
        type_(array_local_type(el_type, len)) { }
  }
}
#endif
