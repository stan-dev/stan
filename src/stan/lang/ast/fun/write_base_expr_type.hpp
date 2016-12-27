#ifndef STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_HPP
#define STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_HPP

#include <stan/lang/ast/base_expr_type.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Write a user-readable version of the specified base expression
     * type to the specified output stream.
     *
     * @param o output stream
     * @param type base expression type
     */
    std::ostream& write_base_expr_type(std::ostream& o, base_expr_type type);

  }
}
#endif
