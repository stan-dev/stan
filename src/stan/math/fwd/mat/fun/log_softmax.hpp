#ifndef STAN__MATH__FWD__MAT__FUN__LOG_SOFTMAX_HPP
#define STAN__MATH__FWD__MAT__FUN__LOG_SOFTMAX_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/softmax.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/log_softmax.hpp>
#include <stan/math/prim/mat/fun/softmax.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    inline
    Eigen::Matrix<fvar<T>,Eigen::Dynamic,1>
    log_softmax(const Eigen::Matrix<fvar<T>,Eigen::Dynamic,1>& alpha) {
      using stan::math::softmax;
      using stan::math::log_softmax;
      using Eigen::Matrix;
      using Eigen::Dynamic;

      Matrix<T,Dynamic,1> alpha_t(alpha.size());
      for (int k = 0; k < alpha.size(); ++k)
        alpha_t(k) = alpha(k).val_;

      Matrix<T,Dynamic,1> softmax_alpha_t = softmax(alpha_t);
      Matrix<T,Dynamic,1> log_softmax_alpha_t = log_softmax(alpha_t);

      Matrix<fvar<T>,Dynamic,1> log_softmax_alpha(alpha.size());
      for (int k = 0; k < alpha.size(); ++k) {
        log_softmax_alpha(k).val_ = log_softmax_alpha_t(k);
        log_softmax_alpha(k).d_ = 0;
      }

      // for each input position
      for (int m = 0; m < alpha.size(); ++m) {
        T negative_alpha_m_d_times_softmax_alpha_t_m
          = - alpha(m).d_ * softmax_alpha_t(m);
        // for each output position
        for (int k = 0; k < alpha.size(); ++k) {
          // chain from input to output
          if (m == k)
            log_softmax_alpha(k).d_
              += alpha(m).d_
              + negative_alpha_m_d_times_softmax_alpha_t_m;
          else
            log_softmax_alpha(k).d_
              += negative_alpha_m_d_times_softmax_alpha_t_m;
        }
      }

      return log_softmax_alpha;
    }


  }
}

#endif
