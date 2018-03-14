#ifndef STAN_LANG_AST_VECTOR_TYPE_DEF_HPP
#define STAN_LANG_AST_VECTOR_TYPE_DEF_HPP

#include <stan/lang/ast/type/vector_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    vector_type::vector_type() : is_data_(false) { }

    vector_type::vector_type(bool is_data) : is_data_(is_data) { }

    std::string vector_type::oid() const {
      return std::string("04_vector_type");
    }
  }
}
#endif

