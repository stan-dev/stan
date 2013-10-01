#ifndef __STAN__MATH__MATRIX__LOG_SUM_EXP_HPP__
#define __STAN__MATH__MATRIX__LOG_SUM_EXP_HPP__

#include <limits>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/math/matrix/Eigen.hpp>


namespace stan {

  namespace math {

    /**
     * Return the log of the sum of the exponentiated values of the specified
     * matrix of values.  The matrix may be a full matrix, a vector,
     * or a row vector.
     *
     * The function is defined as follows to prevent overflow in exponential
     * calculations.
     *
     * \f$\log \sum_{n=1}^N \exp(x_n) = \max(x) + \log \sum_{n=1}^N \exp(x_n - \max(x))\f$.
     * 
     * @param[in] x Matrix of specified values
     * @return The log of the sum of the exponentiated vector values.
     */
    template <typename T, int R, int C>
    T log_sum_exp(const Eigen::Matrix<T,R,C>& x) {
      using std::numeric_limits;
      using std::log;
      using std::exp;
      T max = -numeric_limits<T>::infinity();
      for (int i = 0; i < x.size(); i++) 
        if (x(i) > max) 
          max = x(i);
            
      T sum = 0.0;
      for (int i = 0; i < x.size(); i++) 
        if (x(i) != -numeric_limits<double>::infinity()) 
          sum += exp(x(i) - max);
          
      return max + log(sum);
    }

  }
}

#endif
