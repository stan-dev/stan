#ifndef STAN__MATH__REV__ARR__META__ZERO_ALL_ADJOINTS_HPP
#define STAN__MATH__REV__ARR__META__ZERO_ALL_ADJOINTS_HPP

#include <stan/math/rev/arr/meta/var_stack.hpp>

namespace stan {
  namespace agrad {

    /**
     * Reset all adjoint values in the stack to zero.
     */
    static void set_zero_all_adjoints() {
      for (size_t i = 0; i < ChainableStack::var_stack_.size(); ++i)
        ChainableStack::var_stack_[i]->set_zero_adjoint();
      for (size_t i = 0; i < ChainableStack::var_nochain_stack_.size(); ++i)
        ChainableStack::var_nochain_stack_[i]->set_zero_adjoint();
    }

  }
}
#endif
