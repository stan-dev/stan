#ifndef STAN__MATH__REV__MAT__FUN__SD_HPP
#define STAN__MATH__REV__MAT__FUN__SD_HPP

#include <cmath>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/mean.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/err/check_nonzero_size.hpp>

namespace stan {

  namespace agrad {

    namespace {  // anonymous

      // if x.size() = N, and x[i] = x[j] =
      // then lim sd(x) -> 0 [ d/dx[n] sd(x) ] = sqrt(N) / N

      var calc_sd(size_t size,
                  const var* dtrs) {
        using std::sqrt;
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
        double sd = sqrt(variance);
        double* partials = (double*) ChainableStack::memalloc_.alloc(size * sizeof(double));
        if (sum_of_squares < 1e-20) {
          double grad_limit = 1 / std::sqrt((double)size);
          for (size_t i = 0; i < size; ++i)
            partials[i] = grad_limit;
        } else {
          double multiplier = 1 / (sd * (size - 1));
          for (size_t i = 0; i < size; ++i)
            partials[i] = multiplier * (dtrs[i].vi_->val_ - mean);
        }
        return var(new stored_gradient_vari(sd, size,
                                            varis, partials));
      }

    }

    /**
     * Return the sample standard deviation of the specified standard
     * vector.  Raise domain error if size is not greater than zero.
     *
     * @param[in] v a vector
     * @return sample standard deviation of specified vector
     */
    var sd(const std::vector<var>& v) {
      stan::math::check_nonzero_size("sd", "v", v);
      if (v.size() == 1) return 0;
      return calc_sd(v.size(), &v[0]);
    }

    /*
     * Return the sample standard deviation of the specified vector,
     * row vector, or matrix.  Raise domain error if size is not
     * greater than zero.
     *
     * @tparam R number of rows
     * @tparam C number of columns
     * @param[in] m input matrix
     * @return sample standard deviation of specified matrix
     */
    template <int R, int C>
    var sd(const Eigen::Matrix<var,R,C>& m) {
      stan::math::check_nonzero_size("sd", "m", m);
      if (m.size() == 1) return 0;
      return calc_sd(m.size(), &m(0));
    }

  }
}

#endif
