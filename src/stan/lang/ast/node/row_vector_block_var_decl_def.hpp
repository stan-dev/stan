#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    row_vector_block_var_decl::row_vector_block_var_decl() { }

    row_vector_block_var_decl::row_vector_block_var_decl(
                               const std::string& name,
                               const range& bounds,
                               const expression& N,
                               const expression& def)
      : var_decl(name, bare_expr_type(row_vector_type()), def),
        type_(row_vector_block_type(bounds, N)) { }
  }
}
#endif
