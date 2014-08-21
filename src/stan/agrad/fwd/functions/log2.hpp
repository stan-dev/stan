#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG2_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG2_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/log2.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log2(const fvar<T>& x) {
      using std::log;
      using stan::math::log2;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
        return fvar<T>(log2(x.val_), x.d_ / (x.val_ * stan::math::LOG_2));
    }
  }
}
#endif
