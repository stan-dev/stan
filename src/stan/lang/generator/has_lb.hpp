#ifndef STAN_LANG_GENERATOR_HAS_LB_HPP
#define STAN_LANG_GENERATOR_HAS_LB_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the the specified declaration has a lower bound
     * range constraint but not an upper bound range constraint.
     *
     * @tparam D type of declaration
     * @param[in] x declaration
     * @return true if the declaration has only a lower bound constraint
     */
    template <typename D>
    bool has_lb(const D& x) {
      return !is_nil(x.range_.low_.expr_) && is_nil(x.range_.high_.expr_);
    }

  }
}
#endif
