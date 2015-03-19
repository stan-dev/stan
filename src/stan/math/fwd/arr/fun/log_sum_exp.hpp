#ifndef STAN__MATH__FWD__ARR__FUN__LOG_SUM_EXP_HPP
#define STAN__MATH__FWD__ARR__FUN__LOG_SUM_EXP_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/arr/fun/log_sum_exp.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    fvar<T>
    log_sum_exp(const std::vector<fvar<T> >& v) {
      using stan::math::log_sum_exp;
      using std::exp;
      std::vector<T> vals(v.size());
      for (size_t i = 0; i < v.size(); ++i)
        vals[i] = v[i].val_;
      T deriv(0.0);
      T denominator(0.0);
      for (size_t i = 0; i < v.size(); ++i) {
        T exp_vi = exp(vals[i]);
        denominator += exp_vi;
        deriv += v[i].d_ * exp_vi;
      }
      return fvar<T>(log_sum_exp(vals), deriv / denominator);
    }

  }
}
#endif
