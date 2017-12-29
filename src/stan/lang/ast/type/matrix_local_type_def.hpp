#ifndef STAN_LANG_AST_MATRIX_LOCAL_TYPE_DEF_HPP
#define STAN_LANG_AST_MATRIX_LOCAL_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    matrix_local_type::matrix_local_type() { }

    matrix_local_type::matrix_local_type(const expression& M,
                                         const expression& N)
      : M_(M), N_(N) { }
  }
}
#endif
