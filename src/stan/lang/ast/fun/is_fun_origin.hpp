#ifndef STAN_LANG_AST_FUN_IS_FUN_ORIGIN_HPP
#define STAN_LANG_AST_FUN_IS_FUN_ORIGIN_HPP

#include <stan/lang/ast/var_origin.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the specified variable origin is for variables
     * declared as function arguments.
     *
     * @param vo variable origin
     * @return true if the origin is for function argument variables
     */
    bool is_fun_origin(const var_origin& vo);

  }
}
#endif
