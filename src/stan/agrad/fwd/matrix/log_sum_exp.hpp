#ifndef __STAN__AGRAD__FWD__MATRIX__LOG__SUM__EXP__HPP__
#define __STAN__AGRAD__FWD__MATRIX__LOG__SUM__EXP__HPP__

#include <vector>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan{

  namespace agrad{

    // FIXME: cut-and-paste from fwd/log_sum_exp.hpp; should
    // be able to generalize
    template <typename T, int R, int C>
    fvar<T>
    log_sum_exp(const Eigen::Matrix<T,R,C>& v) {
      using stan::math::log_sum_exp;
      using std::exp;
      std::vector<T> vals(v.size());
      for (int i = 0; i < v.size(); ++i)
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
