#ifndef STAN__AGRAD__FWD__FUNCTIONS__ABS_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ABS_HPP

#include <stan/agrad/fwd/functions/value_of.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/abs.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/likely.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline
    fvar<T>
    abs(const fvar<T>& x) {
      using stan::math::abs;
      using stan::math::value_of;
      if (x.val_ > 0.0)
        return x;
      else if (x.val_ < 0.0)
        return fvar<T>(-x.val_, -x.d_);
      else if (x.val_ == 0.0)
        return fvar<T>(0, 0);
      else // if (unlikely(boost::math::isnan(value_of(x.val_)))) 
        return fvar<T>(abs(x.val_),stan::math::NOT_A_NUMBER);
    }
  }
}
#endif
