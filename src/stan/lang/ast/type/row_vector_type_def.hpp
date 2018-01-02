#ifndef STAN_LANG_AST_ROW_VECTOR_TYPE_DEF_HPP
#define STAN_LANG_AST_ROW_VECTOR_TYPE_DEF_HPP

#include <string>

namespace stan {
  namespace lang {

    std::string row_vector_type::oid() const {
      return std::string("05_row_vector_type");
    }
  }
}
#endif

