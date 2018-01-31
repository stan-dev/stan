#ifndef STAN_LANG_AST_VECTOR_LOCAL_TYPE_DEF_HPP
#define STAN_LANG_AST_VECTOR_LOCAL_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    vector_local_type::vector_local_type() { }

    vector_local_type::vector_local_type(const expression& N)
      : N_(N) { }

    expression vector_local_type::N() const { return N_; }
  }
}
#endif
