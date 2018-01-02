#ifndef STAN_LANG_AST_ILL_FORMED_TYPE_DEF_HPP
#define STAN_LANG_AST_ILL_FORMED_TYPE_DEF_HPP

#include <string>

namespace stan {
  namespace lang {

    std::string ill_formed_type::oid() const {
      return std::string("00_ill_formed_type");
    }
  }
}
#endif

