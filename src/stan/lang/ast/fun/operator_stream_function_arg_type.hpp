#ifndef STAN_LANG_AST_FUN_OPERATOR_STREAM_FUNCTION_ARG_TYPE_HPP
#define STAN_LANG_AST_FUN_OPERATOR_STREAM_FUNCTION_ARG_TYPE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct function_arg_type;

    /**
     * Stream a user-readable version of the function argument type to the
     * specified output stream, returning the speicifed argument
     * output stream to allow chaining.
     *
     * @param o output stream
     * @param et function argument type
     * @return argument output stream
     */
    std::ostream& operator<<(std::ostream& o,
                             const function_arg_type& fa_type);

  }
}
#endif
