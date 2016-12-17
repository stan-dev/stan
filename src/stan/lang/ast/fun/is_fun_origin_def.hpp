#ifndef STAN_LANG_AST_FUN_IS_FUN_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_IS_FUN_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bool is_fun_origin(const var_origin& vo) {
      return vo == function_argument_origin
        || vo == function_argument_origin_lp
        || vo == function_argument_origin_rng
        || vo == void_function_argument_origin
        || vo == void_function_argument_origin_lp
        || vo == void_function_argument_origin_rng;
    }

  }
}
#endif
