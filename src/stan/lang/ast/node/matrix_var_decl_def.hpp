#ifndef STAN_LANG_AST_NODE_MATRIX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    matrix_var_decl::matrix_var_decl() : base_var_decl(MATRIX_T) { }

    matrix_var_decl::matrix_var_decl(const range& range, const expression& M,
                                     const expression& N,
                                     const std::string& name,
                                     const std::vector<expression>& dims,
                                     const expression& def)
      : base_var_decl(name, dims, MATRIX_T, def), range_(range), M_(M), N_(N) {
    }

  }
}
#endif
