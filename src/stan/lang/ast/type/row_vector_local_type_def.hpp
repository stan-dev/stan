#ifndef STAN_LANG_AST_ROW_VECTOR_LOCAL_TYPE_DEF_HPP
#define STAN_LANG_AST_ROW_VECTOR_LOCAL_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    row_vector_local_type::row_vector_local_type() { }

    row_vector_local_type::row_vector_local_type(const expression& N)
      : N_(N) { }
  }
}
#endif
