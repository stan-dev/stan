#ifndef STAN__MATH__FWD__SCAL__FUN__ABS_HPP
#define STAN__MATH__FWD__SCAL__FUN__ABS_HPP

#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/abs.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/likely.hpp>

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
