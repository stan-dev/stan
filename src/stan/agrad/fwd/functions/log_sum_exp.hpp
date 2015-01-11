#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG_SUM_EXP_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG_SUM_EXP_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_sum_exp.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log_sum_exp(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<T>(log_sum_exp(x1.val_, x2.val_),
                     x1.d_ / (1 + exp(x2.val_ - x1.val_))
                     + x2.d_ / (exp(x1.val_ - x2.val_) + 1));
    }

    template <typename T>
    inline
    fvar<T>
    log_sum_exp(const double x1, const fvar<T>& x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<T>(log_sum_exp(x1, x2.val_),
                     x2.d_ / (exp(x1 - x2.val_) + 1));
    }

    template <typename T>
    inline
    fvar<T>
    log_sum_exp(const fvar<T>& x1, const double x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<T>(log_sum_exp(x1.val_, x2),
                     x1.d_ / (1 + exp(x2 - x1.val_)));
    }

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
