#ifndef STAN__MATH__PRIM__ARR__FUN__LOG_SUM_EXP_HPP
#define STAN__MATH__PRIM__ARR__FUN__LOG_SUM_EXP_HPP

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the log of the sum of the exponentiated values of the specified
     * sequence of values.
     *
     * The function is defined as follows to prevent overflow in exponential
     * calculations.
     *
     * \f$\log \sum_{n=1}^N \exp(x_n) = \max(x) + \log \sum_{n=1}^N \exp(x_n - \max(x))\f$.
     *
     * @param[in] x array of specified values
     * @return The log of the sum of the exponentiated vector values.
     */
    double log_sum_exp(const std::vector<double>& x) {
      using std::numeric_limits;
      using std::log;
      using std::exp;
      double max = -numeric_limits<double>::infinity();
      for (size_t ii = 0; ii < x.size(); ii++)
        if (x[ii] > max)
          max = x[ii];
      double sum = 0.0;
      for (size_t ii = 0; ii < x.size(); ii++)
        if (x[ii] != -numeric_limits<double>::infinity())
          sum += exp(x[ii] - max);

      return max + log(sum);
    }

  }
}

#endif
