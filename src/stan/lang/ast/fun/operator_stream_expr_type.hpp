#ifndef STAN_LANG_AST_FUN_OPERATOR_STREAM_EXPR_TYPE_HPP
#define STAN_LANG_AST_FUN_OPERATOR_STREAM_EXPR_TYPE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct expr_type;

    /**
     * Stream a user-readable version of the expression type to the
     * specified output stream, returning the speicifed argument
     * output stream to allow chaining.
     *
     * @param o output stream
     * @param et expression type
     * @return argument output stream
     */
    std::ostream& operator<<(std::ostream& o, const expr_type& et);

  }
}
#endif
