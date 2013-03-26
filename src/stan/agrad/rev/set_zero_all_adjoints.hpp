#ifndef __STAN__AGRAD__REV__SET_ZERO_ALL_ADJOINTS_HPP__
#define __STAN__AGRAD__REV__SET_ZERO_ALL_ADJOINTS_HPP__

#include <stan/agrad/var_stack.hpp>

namespace stan {
  namespace agrad {

    /**
     * Reset all adjoint values in the stack to zero.
     */
    static void set_zero_all_adjoints() {
      for (size_t i = 0; i < var_stack_.size(); ++i)
        var_stack_[i]->set_zero_adjoint();
      for (size_t i = 0; i < var_nochain_stack_.size(); ++i)
        var_nochain_stack_[i]->set_zero_adjoint();
    }

  }
}
#endif
