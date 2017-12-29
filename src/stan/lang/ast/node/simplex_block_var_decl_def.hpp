#ifndef STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    simplex_block_var_decl::simplex_block_var_decl() { }

    simplex_block_var_decl::simplex_block_var_decl(
                            const std::string& name,
                            const expression& K,
                            const expression& def)
      : var_decl(name, bare_expr_type(vector_type()), def),
        type_(simplex_block_type(K)) { }
  }
}
#endif
