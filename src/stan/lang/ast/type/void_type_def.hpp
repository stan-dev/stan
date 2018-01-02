#ifndef STAN_LANG_AST_VOID_TYPE_DEF_HPP
#define STAN_LANG_AST_VOID_TYPE_DEF_HPP

#include <string>

namespace stan {
  namespace lang {

    std::string void_type::oid() const {
      return std::string("01_void_type");
    }
  }
}
#endif

