#ifndef STAN__AGRAD__REV__MATRIX__LOG_SOFTMAX_HPP
#define STAN__AGRAD__REV__MATRIX__LOG_SOFTMAX_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_softmax.hpp>
#include <stan/math/matrix/softmax.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class log_softmax_elt_vari : public vari {
      private:
        vari** alpha_;
        const double* softmax_alpha_;
        const int size_;  // array sizes
        const int idx_;   // in in softmax output
      public:
        log_softmax_elt_vari(double val, 
                             vari** alpha,
                             const double* softmax_alpha,
                             int size,
                             int idx)
          : vari(val),
            alpha_(alpha),
            softmax_alpha_(softmax_alpha),
            size_(size),
            idx_(idx) {
        }
        void chain() {
          for (int m = 0; m < size_; ++m) {
            if (m == idx_)
              alpha_[m]->adj_ +=  adj_ * (1 - softmax_alpha_[m]);
            else
              alpha_[m]->adj_ -= adj_ * softmax_alpha_[m];
          }
        }
      };

    }


    /**
     * Return the softmax of the specified Eigen vector.  Softmax is
     * guaranteed to return a simplex.
     *
     * The gradient calculations are unfolded.
     * 
     * @param alpha Unconstrained input vector.
     * @return Softmax of the input.
     * @throw std::domain_error If the input vector is size 0.
     */
    inline Eigen::Matrix<var,Eigen::Dynamic,1> 
    log_softmax(const Eigen::Matrix<var,Eigen::Dynamic,1>& alpha) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      stan::error_handling::check_nonzero_size("log_softmax", "alpha", alpha);

      if (alpha.size() == 0) 
        throw std::domain_error("arg vector to log_softmax() must have size > 0");
      if (alpha.size() == 0) 
        throw std::domain_error("arg vector to log_softmax() must have size > 0");

      if (alpha.size() == 0) 
        throw std::domain_error("arg vector to log_softmax() must have size > 0");

      vari** alpha_vi_array 
        = (vari**) agrad::chainable::operator new(sizeof(vari*) * alpha.size());
      for (int i = 0; i < alpha.size(); ++i)
        alpha_vi_array[i] = alpha(i).vi_;
      

      Matrix<double,Dynamic,1> alpha_d(alpha.size());
      for (int i = 0; i < alpha_d.size(); ++i)
        alpha_d(i) = alpha(i).val();
      
      // fold logic of math::softmax() and math::log_softmax() to save computations

      Matrix<double,Dynamic,1> softmax_alpha_d(alpha_d.size());
      Matrix<double,Dynamic,1> log_softmax_alpha_d(alpha_d.size());

      double max_v = alpha_d.maxCoeff();

      double sum = 0.0;
      for (int i = 0; i < alpha_d.size(); ++i) {
        softmax_alpha_d(i) = std::exp(alpha_d(i) - max_v);
        sum += softmax_alpha_d(i);
      }

      for (int i = 0; i < alpha_d.size(); ++i)
        softmax_alpha_d(i) /= sum;
      double log_sum = std::log(sum);

      for (int i = 0; i < alpha_d.size(); ++i)
        log_softmax_alpha_d(i) = (alpha_d(i) - max_v) - log_sum;

      // end fold

      double* softmax_alpha_d_array 
         = (double*) agrad::chainable::operator new(sizeof(double) * alpha_d.size());

      for (int i = 0; i < alpha_d.size(); ++i)
        softmax_alpha_d_array[i] = softmax_alpha_d(i);

      Matrix<var,Dynamic,1> log_softmax_alpha(alpha.size());
      for (int k = 0; k < log_softmax_alpha.size(); ++k)
        log_softmax_alpha(k) = var(new log_softmax_elt_vari(log_softmax_alpha_d[k],
                                                            alpha_vi_array,
                                                            softmax_alpha_d_array,
                                                            alpha.size(),
                                                            k));
      return log_softmax_alpha;
    }
    

  }
}

#endif
