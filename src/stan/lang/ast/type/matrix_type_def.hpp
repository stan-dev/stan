#ifndef STAN_LANG_AST_MATRIX_TYPE_DEF_HPP
#define STAN_LANG_AST_MATRIX_TYPE_DEF_HPP

#include <stan/lang/ast/type/matrix_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_type::matrix_type() : is_data_(false) { }

    matrix_type::matrix_type(bool is_data) : is_data_(is_data) { }

    std::string matrix_type::oid() const {
      return std::string("06_matrix_type");
    }
  }
}
#endif

