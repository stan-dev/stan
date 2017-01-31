#ifndef STAN_LANG_AST_FUN_IS_FUN_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_IS_FUN_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bool is_fun_origin(const var_origin& vo) {
      return vo.program_block_ == function_argument_origin
        || vo.program_block_ == function_argument_origin_lp
        || vo.program_block_ == function_argument_origin_rng
        || vo.program_block_ == void_function_argument_origin
        || vo.program_block_ == void_function_argument_origin_lp
        || vo.program_block_ == void_function_argument_origin_rng;
    }

  }
}
#endif
