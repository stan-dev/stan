#ifndef STAN__AGRAD__FWD__MATRIX__LOG_SUM_EXP_HPP
#define STAN__AGRAD__FWD__MATRIX__LOG_SUM_EXP_HPP

#include <vector>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix/log_sum_exp.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/agrad/fwd/functions/exp.hpp>

namespace stan{

  namespace agrad{

    // FIXME: cut-and-paste from fwd/log_sum_exp.hpp; should
    // be able to generalize
    template <typename T, int R, int C>
    fvar<T>
    log_sum_exp(const Eigen::Matrix<fvar<T>,R,C>& v) {
      using stan::math::log_sum_exp;
      using stan::agrad::log_sum_exp;
      using stan::agrad::exp;
      using std::exp;
      using stan::agrad::log;
      using std::log;

      Eigen::Matrix<T,1,Eigen::Dynamic> vals(v.size());
      for (int i = 0; i < v.size(); ++i)
        vals[i] = v(i).val_;
      T deriv(0.0);
      T denominator(0.0);
      for (size_t i = 0; i < v.size(); ++i) {
        T exp_vi = exp(vals[i]);
        denominator += exp_vi;
        deriv += v(i).d_ * exp_vi;
      }
      return fvar<T>(log_sum_exp(vals), deriv / denominator);
    }
    
  }
}
#endif
