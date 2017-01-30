#ifndef STAN_LANG_AST_FUN_IS_DATA_ORIGIN_HPP
#define STAN_LANG_AST_FUN_IS_DATA_ORIGIN_HPP

#include <stan/lang/ast/var_origin.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the specified variable origin is for variables
     * declared as data or transformed data.
     *
     * @param vo variable origin
     * @return true if origin for variables declared as data
     */
    bool is_data_origin(const var_origin& vo);

  }
}
#endif
