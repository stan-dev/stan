#ifndef STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    ordered_block_var_decl::ordered_block_var_decl() { }

    ordered_block_var_decl::ordered_block_var_decl(const std::string& name,
                                                   const expression& K,
                                                   const expression& def)
      : var_decl(name, bare_expr_type(vector_type()), def),
        type_(ordered_block_type(K)) { }
  }
}
#endif
