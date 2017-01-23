#ifndef STAN_LANG_GENERATOR_HAS_LUB_HPP
#define STAN_LANG_GENERATOR_HAS_LUB_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the the specified declaration has a lower bound
     * and upper bound range constraint.
     *
     * @tparam D type of declaration
     * @param[in] x declaration
     * @return true if the declaration has lower and upper bounds
     */
    template <typename D>
    bool has_lub(const D& x) {
      return !is_nil(x.range_.low_.expr_) && !is_nil(x.range_.high_.expr_);
    }

  }
}
#endif
