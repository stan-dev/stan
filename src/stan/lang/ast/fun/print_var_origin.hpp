#ifndef STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_HPP
#define STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_HPP

#include <stan/lang/ast/var_origin.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Write a user-readable version of the specified variable to
     * origin to the specified output stream.
     *
     * @param o output stream
     * @param vo variable origin
     */
    void print_var_origin(std::ostream& o, const var_origin& vo);

  }
}
#endif
