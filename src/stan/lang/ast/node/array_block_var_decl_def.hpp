#ifndef STAN_LANG_AST_NODE_ARRAY_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ARRAY_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    array_block_var_decl::array_block_var_decl() { }

    array_block_var_decl::array_block_var_decl(
                          const std::string& name,
                          const block_array_type& type)
      : var_decl(name, bare_array_type(type.element_type().bare_type())),
        type_(type.element_type(), type.array_len()) { }

    array_block_var_decl::array_block_var_decl(
                          const std::string& name,
                          const block_array_type& type,
                          const expression& def)
      : var_decl(name, bare_array_type(type.element_type().bare_type()), def),
        type_(type.element_type(), type.array_len()) { }
  }
}
#endif
