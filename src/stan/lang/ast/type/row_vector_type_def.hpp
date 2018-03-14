#ifndef STAN_LANG_AST_ROW_VECTOR_TYPE_DEF_HPP
#define STAN_LANG_AST_ROW_VECTOR_TYPE_DEF_HPP

#include <stan/lang/ast/type/row_vector_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    row_vector_type::row_vector_type() : is_data_(false) { }

    row_vector_type::row_vector_type(bool is_data) : is_data_(is_data) { }

    std::string row_vector_type::oid() const {
      return std::string("05_row_vector_type");
    }
  }
}
#endif
