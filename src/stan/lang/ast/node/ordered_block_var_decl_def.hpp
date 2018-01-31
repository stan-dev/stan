#ifndef STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    ordered_block_var_decl::ordered_block_var_decl() { }

    ordered_block_var_decl::ordered_block_var_decl(const std::string& name,
                                                   const ordered_block_type& type)
      : var_decl(name, vector_type()), type_(type.K()) { }

    ordered_block_var_decl::ordered_block_var_decl(const std::string& name,
                                                   const ordered_block_type& type,
                                                   const expression& def)
      : var_decl(name, vector_type(), def), type_(type.K()) { }
  }
}
#endif
