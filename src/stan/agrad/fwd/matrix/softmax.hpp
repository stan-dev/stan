#ifndef STAN__AGRAD__FWD__MATRIX__SOFTMAX_HPP
#define STAN__AGRAD__FWD__MATRIX__SOFTMAX_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/softmax.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    inline 
    Eigen::Matrix<fvar<T>,Eigen::Dynamic,1>
    softmax(const Eigen::Matrix<fvar<T>,Eigen::Dynamic,1>& alpha) {
      using stan::math::softmax;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      
      Matrix<T,Dynamic,1> alpha_t(alpha.size());
      for (int k = 0; k < alpha.size(); ++k)
        alpha_t(k) = alpha(k).val_;
      
      Matrix<T,Dynamic,1> softmax_alpha_t = softmax(alpha_t);

      Matrix<fvar<T>,Dynamic,1> softmax_alpha(alpha.size());
      for (int k = 0; k < alpha.size(); ++k) {
        softmax_alpha(k).val_ = softmax_alpha_t(k);
        softmax_alpha(k).d_ = 0;
      }

      // for each input position
      for (int m = 0; m < alpha.size(); ++m) {
        // for each output position
        T negative_alpha_m_d_times_softmax_alpha_t_m
          = - alpha(m).d_ * softmax_alpha_t(m);
        for (int k = 0; k < alpha.size(); ++k) {
          // chain from input to output
          if (m == k) {
            softmax_alpha(k).d_ 
              += softmax_alpha_t(k) 
              * (alpha(m).d_ 
                 + negative_alpha_m_d_times_softmax_alpha_t_m);
          } else {
            softmax_alpha(k).d_ 
              += negative_alpha_m_d_times_softmax_alpha_t_m
              * softmax_alpha_t(k);
          }
        }
      }

      return softmax_alpha;
    }


  }
}

#endif
