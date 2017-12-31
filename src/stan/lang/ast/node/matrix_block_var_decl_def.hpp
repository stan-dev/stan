#ifndef STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_block_var_decl::matrix_block_var_decl() { }

    matrix_block_var_decl::matrix_block_var_decl(const std::string& name,
                                                 const range& bounds,
                                                 const expression& M,
                                                 const expression& N)
      : var_decl(name, bare_expr_type(matrix_type())),
        type_(matrix_block_type(bounds, M, N)) { }

    matrix_block_var_decl::matrix_block_var_decl(const std::string& name,
                                                 const range& bounds,
                                                 const expression& M,
                                                 const expression& N,
                                                 const expression& def)
      : var_decl(name, bare_expr_type(matrix_type()), def),
        type_(matrix_block_type(bounds, M, N)) { }
  }
}
#endif
