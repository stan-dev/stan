#ifndef STAN_LANG_AST_NODE_CHOLESKY_FACTOR_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_CHOLESKY_FACTOR_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    cholesky_factor_var_decl::cholesky_factor_var_decl()
      : base_var_decl(MATRIX_T) { }

    cholesky_factor_var_decl::cholesky_factor_var_decl(expression const& M,
                                       expression const& N,
                                       std::string const& name,
                                       std::vector<expression> const& dims,
                                       expression const& def)

      : base_var_decl(name, dims, MATRIX_T, def), M_(M), N_(N) { }

  }
}
#endif
