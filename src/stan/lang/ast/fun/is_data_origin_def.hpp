#ifndef STAN_LANG_AST_FUN_IS_DATA_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_IS_DATA_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bool is_data_origin(const var_origin& vo) {
      return vo.program_block_ == data_origin
        || vo.program_block_ == transformed_data_origin;
    }

  }
}
#endif
