#ifndef __STAN__AGRAD__REV__IS_UNINITIALIZED_HPP__
#define __STAN__AGRAD__REV__IS_UNINITIALIZED_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/operator_unary_negative.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    inline bool is_uninitialized(T x) {
      return false;
    }
    inline bool is_uninitialized(var x) {
      return x.is_uninitialized();
    }

  }
}
#endif
