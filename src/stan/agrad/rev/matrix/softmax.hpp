#ifndef STAN__AGRAD__REV__MATRIX__SOFTMAX_HPP
#define STAN__AGRAD__REV__MATRIX__SOFTMAX_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/softmax.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class softmax_elt_vari : public vari {
      private:
        vari** alpha_;
        const double* softmax_alpha_;
        const int size_;  // array sizes
        const int idx_;   // in in softmax output
      public:
        softmax_elt_vari(double val, 
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
            if (m == idx_) {
              alpha_[m]->adj_ 
                +=  adj_ * softmax_alpha_[idx_] * (1 - softmax_alpha_[m]);
            } else {
              alpha_[m]->adj_ 
                -= adj_ * softmax_alpha_[idx_] * softmax_alpha_[m];
            }
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
    softmax(const Eigen::Matrix<var,Eigen::Dynamic,1>& alpha) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      stan::error_handling::check_nonzero_size("softmax", "alpha", alpha);

      vari** alpha_vi_array
        = (vari**) ChainableStack::memalloc_.alloc(sizeof(vari*) * alpha.size());
      for (int i = 0; i < alpha.size(); ++i)
        alpha_vi_array[i] = alpha(i).vi_;
      
      Matrix<double,Dynamic,1> alpha_d(alpha.size());
      for (int i = 0; i < alpha_d.size(); ++i)
        alpha_d(i) = alpha(i).val();
      
      Matrix<double,Dynamic,1> softmax_alpha_d
        = stan::math::softmax(alpha_d);

      double* softmax_alpha_d_array 
        = (double*) ChainableStack::memalloc_.alloc(sizeof(double) * alpha_d.size());
      for (int i = 0; i < alpha_d.size(); ++i)
        softmax_alpha_d_array[i] = softmax_alpha_d(i);

      Matrix<var,Dynamic,1> softmax_alpha(alpha.size());
      for (int k = 0; k < softmax_alpha.size(); ++k)
        softmax_alpha(k) = var(new softmax_elt_vari(softmax_alpha_d[k],
                                                    alpha_vi_array,
                                                    softmax_alpha_d_array,
                                                    alpha.size(),
                                                    k));
      return softmax_alpha;
    }
    

  }
}

#endif
