#ifndef STAN_LANG_AST_DOUBLE_TYPE_DEF_HPP
#define STAN_LANG_AST_DOUBLE_TYPE_DEF_HPP

#include <stan/lang/ast/type/double_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    std::string double_type::oid() const {
      return std::string("03_double_type");
    }
  }
}
#endif

