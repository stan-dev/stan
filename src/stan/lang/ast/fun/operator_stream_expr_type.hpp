#ifndef STAN_LANG_AST_FUN_OPERATOR_STREAM_EXPR_TYPE_HPP
#define STAN_LANG_AST_FUN_OPERATOR_STREAM_EXPR_TYPE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct expr_type;

    std::ostream& operator<<(std::ostream& o, const expr_type& et);

  }
}
#endif
