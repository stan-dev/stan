#ifndef STAN__MATH__REV__CORE__CHAINABLESTACK_HPP
#define STAN__MATH__REV__CORE__CHAINABLESTACK_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    // forward declaration of chainable
    class chainable;
    class chainable_alloc;

    typedef AutodiffStackStorage<chainable,chainable_alloc> ChainableStack;

  }
}
#endif
