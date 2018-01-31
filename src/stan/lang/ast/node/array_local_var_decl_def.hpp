#ifndef STAN_LANG_AST_NODE_LOCAL_ARRAY_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_LOCAL_ARRAY_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    array_local_var_decl::array_local_var_decl() { }

    array_local_var_decl::array_local_var_decl(
                          const std::string& name,
                          const local_array_type& type)
      : var_decl(name, bare_array_type(type.element_type().bare_type())),
        type_(type.element_type(), type.array_len()) { }

    array_local_var_decl::array_local_var_decl(
                          const std::string& name,
                          const local_array_type& type,
                          const expression& def)
      : var_decl(name, bare_array_type(type.element_type().bare_type()), def),
        type_(type.element_type(), type.array_len()) { }
  }
}
#endif
