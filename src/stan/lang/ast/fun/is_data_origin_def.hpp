#ifndef STAN_LANG_AST_FUN_IS_DATA_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_IS_DATA_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bool is_data_origin(const var_origin& vo) {
      return vo == data_origin || vo == transformed_data_origin;
    }

  }
}
#endif
