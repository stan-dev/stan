#ifndef STAN__AGRAD__REV__MATRIX__VARIANCE_HPP
#define STAN__AGRAD__REV__MATRIX__VARIANCE_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/mean.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/stored_gradient_vari.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {

  namespace agrad {
    
    namespace {  // anonymous

      var calc_variance(size_t size,
                        const var* dtrs) {
        vari** varis = (vari**) ChainableStack::memalloc_.alloc(size * sizeof(vari*));
        for (size_t i = 0; i < size; ++i)
          varis[i] = dtrs[i].vi_;
        double sum = 0.0;
        for (size_t i = 0; i < size; ++i)
          sum += dtrs[i].vi_->val_;
        double mean = sum / size;
        double sum_of_squares = 0;
        for (size_t i = 0; i < size; ++i) {
          double diff = dtrs[i].vi_->val_ - mean;
          sum_of_squares += diff * diff;
        }
        double variance = sum_of_squares / (size - 1);
        double* partials = (double*) ChainableStack::memalloc_.alloc(size * sizeof(double));
        double two_over_size_m1 = 2 / (size - 1);
        for (size_t i = 0; i < size; ++i)
          partials[i] = two_over_size_m1 * (dtrs[i].vi_->val_ - mean);
        return var(new stored_gradient_vari(variance, size,
                                            varis, partials));
      }

    }

    /**
     * Return the sample variance of the specified standard
     * vector.  Raise domain error if size is not greater than zero.
     *
     * @param[in] v a vector
     * @return sample variance of specified vector
     */
    var variance(const std::vector<var>& v) {
      stan::error_handling::check_nonzero_size("variance", "v", v);
      if (v.size() == 1) return 0;
      return calc_variance(v.size(), &v[0]);
    }

    /*
     * Return the sample variance of the specified vector, row vector,
     * or matrix.  Raise domain error if size is not greater than
     * zero.
     *
     * @tparam R number of rows
     * @tparam C number of columns
     * @param[in] m input matrix
     * @return sample variance of specified matrix
     */
    template <int R, int C>
    var variance(const Eigen::Matrix<var,R,C>& m) {
      stan::error_handling::check_nonzero_size("variance", "m", m);
      if (m.size() == 1) return 0;
      return calc_variance(m.size(), &m(0));
    }

  }
}

#endif
