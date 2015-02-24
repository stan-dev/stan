#ifndef STAN__MATH__REV__CORE__NESTED_SIZE_HPP
#define STAN__MATH__REV__CORE__NESTED_SIZE_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <cstdlib>

namespace stan {
  namespace agrad {

    static inline size_t nested_size() {
      return ChainableStack::var_stack_.size() - ChainableStack::nested_var_stack_sizes_.back();
    }

  }
}
#endif
