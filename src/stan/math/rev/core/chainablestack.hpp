#ifndef STAN_MATH_REV_CORE_CHAINABLESTACK_HPP
#define STAN_MATH_REV_CORE_CHAINABLESTACK_HPP

#include <stan/math/rev/core/autodiffstackstorage.hpp>

namespace stan {
  namespace math {

    // forward declaration of chainable
    class chainable;
    class chainable_alloc;

    typedef AutodiffStackStorage<chainable, chainable_alloc> ChainableStack;

  }
}
#endif
