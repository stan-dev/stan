#ifndef STAN_LANG_AST_NODE_MATRIX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    matrix_var_decl::matrix_var_decl() : base_var_decl(MATRIX_T) { }

    matrix_var_decl::matrix_var_decl(range const& range, expression const& M,
                                     expression const& N,
                                     std::string const& name,
                                     std::vector<expression> const& dims,
                                     expression const& def)
      : base_var_decl(name, dims, MATRIX_T, def), range_(range), M_(M), N_(N) {
    }

  }
}
#endif
