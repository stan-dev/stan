#ifndef STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    int_block_var_decl::int_block_var_decl() { }

    int_block_var_decl::int_block_var_decl(const std::string& name,
                                           const int_block_type& type)
      : var_decl(name, int_type()), type_(type.bounds()) { }

    int_block_var_decl::int_block_var_decl(const std::string& name,
                                           const int_block_type& type,
                                           const expression& def)
      : var_decl(name, int_type(), def), type_(type.bounds()) { }
  }
}
#endif
