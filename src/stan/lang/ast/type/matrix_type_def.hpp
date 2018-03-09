#ifndef STAN_LANG_AST_MATRIX_TYPE_DEF_HPP
#define STAN_LANG_AST_MATRIX_TYPE_DEF_HPP

#include <stan/lang/ast/type/matrix_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    std::string matrix_type::oid() const {
      return std::string("06_matrix_type");
    }
  }
}
#endif

