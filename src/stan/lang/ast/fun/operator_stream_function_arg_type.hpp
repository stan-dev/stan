#ifndef STAN_LANG_AST_FUN_OPERATOR_STREAM_FUNCTION_ARG_TYPE_HPP
#define STAN_LANG_AST_FUN_OPERATOR_STREAM_FUNCTION_ARG_TYPE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct function_arg_type;

    std::ostream& operator<<(std::ostream& o,
                             const function_arg_type& fa_type);

  }
}
#endif
