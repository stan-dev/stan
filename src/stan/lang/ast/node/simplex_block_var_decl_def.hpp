#ifndef STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    simplex_block_var_decl::simplex_block_var_decl() { }

    simplex_block_var_decl::simplex_block_var_decl(
                            const std::string& name,
                            const simplex_block_type& type)
      : var_decl(name, vector_type()), type_(type.K()) { }

    simplex_block_var_decl::simplex_block_var_decl(
                            const std::string& name,
                            const simplex_block_type& type,
                            const expression& def)
      : var_decl(name, vector_type(), def), type_(type.K()) { }
  }
}
#endif
