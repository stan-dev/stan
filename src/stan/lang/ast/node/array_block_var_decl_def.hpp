#ifndef STAN_LANG_AST_NODE_BLOCK_ARRAY_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_BLOCK_ARRAY_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    array_block_var_decl::array_block_var_decl() { }

    array_block_var_decl::array_block_var_decl(
                          const std::string& name,
                          const block_var_type& el_type,
                          const expression& len)
      : var_decl(name, bare_array_type(el_type.bare_type())),
        type_(block_array_type(el_type, len)) { }

    array_block_var_decl::array_block_var_decl(
                          const std::string& name,
                          const block_var_type& el_type,
                          const expression& len,
                          const expression& def)
      : var_decl(name, bare_array_type(el_type.bare_type()), def),
        type_(block_array_type(el_type, len)) { }
  }
}
#endif
