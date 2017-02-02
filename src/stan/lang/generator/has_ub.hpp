#ifndef STAN_LANG_GENERATOR_HAS_UB_HPP
#define STAN_LANG_GENERATOR_HAS_UB_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the the specified declaration has an upper bound
     * range constraint but not a lower bound range constraint.
     *
     * @tparam D type of declaration
     * @param[in] x declaration
     * @return true if the declaration has only an upper bound constraint
     */
    template <typename D>
    bool has_ub(const D& x) {
      return is_nil(x.range_.low_.expr_) && !is_nil(x.range_.high_.expr_);
    }

  }
}
#endif
