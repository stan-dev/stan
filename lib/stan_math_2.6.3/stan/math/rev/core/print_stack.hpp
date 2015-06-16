#ifndef STAN_MATH_REV_CORE_PRINT_STACK_HPP
#define STAN_MATH_REV_CORE_PRINT_STACK_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/vari.hpp>
#include <ostream>

namespace stan {
  namespace math {

    /**
     * Prints the auto-dif variable stack. This function
     * is used for debugging purposes.
     *
     * Only works if all members of stack are vari* as it
     * casts to vari*.
     *
     * @param o ostream to modify
     */
    inline void print_stack(std::ostream& o) {
      o << "STACK, size=" << ChainableStack::var_stack_.size() << std::endl;
      for (size_t i = 0; i < ChainableStack::var_stack_.size(); ++i)
        o << i
          << "  " << ChainableStack::var_stack_[i]
          << "  " << (static_cast<vari*>(ChainableStack::var_stack_[i]))->val_
          << " : " << (static_cast<vari*>(ChainableStack::var_stack_[i]))->adj_
          << std::endl;
    }

  }
}
#endif
