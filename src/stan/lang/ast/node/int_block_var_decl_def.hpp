#ifndef STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    int_block_var_decl::int_block_var_decl() { }

    int_block_var_decl::int_block_var_decl(const std::string& name,
                                           const range& bounds,
                                           const expression& def)
      : var_decl(name, bare_expr_type(int_type()), def),
        type_(int_block_type(bounds)) { }
  }
}
#endif
