#ifndef STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    vector_block_var_decl::vector_block_var_decl() { }

    vector_block_var_decl::vector_block_var_decl(const std::string& name,
                                                 const range& bounds,
                                                 const expression& N)
      : var_decl(name, bare_expr_type(vector_type())),
        type_(vector_block_type(bounds, N)) { }

    vector_block_var_decl::vector_block_var_decl(const std::string& name,
                                                 const range& bounds,
                                                 const expression& N,
                                                 const expression& def)
      : var_decl(name, bare_expr_type(vector_type()), def),
        type_(vector_block_type(bounds, N)) { }
  }
}
#endif
