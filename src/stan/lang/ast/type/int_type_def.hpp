#ifndef STAN_LANG_AST_INT_TYPE_DEF_HPP
#define STAN_LANG_AST_INT_TYPE_DEF_HPP

#include <stan/lang/ast/type/int_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    std::string int_type::oid() const {
      return std::string("02_int_type");
    }
  }
}
#endif

