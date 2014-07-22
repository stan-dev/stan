#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG1M_EXP_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG1M_EXP_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1m_exp.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    log1m_exp(const fvar<T>& x) {
      using stan::math::log1m_exp;
      using boost::math::expm1;
      using stan::math::NOT_A_NUMBER;
      using std::exp;
      if (x.val_ >= 0)
        return fvar<T>(NOT_A_NUMBER);
      return fvar<T>(log1m_exp(x.val_), x.d_ / -expm1(-x.val_));
    }
  }
}
#endif
