#ifndef __STAN__MATH__FUNCTIONS__LOG_SUM_EXP_HPP__
#define __STAN__MATH__FUNCTIONS__LOG_SUM_EXP_HPP__

#include <stan/math/functions/log1p.hpp>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <limits>

namespace stan {
  namespace math {

    /**
     * Calculates the log sum of exponetials without overflow.
     *
     * \f$\log (\exp(a) + \exp(b)) = m + \log(\exp(a-m) + \exp(b-m))\f$,
     *
     * where \f$m = max(a,b)\f$.
     * 
     * @param a the first variable
     * @param b the second variable
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    log_sum_exp(const T2& a, const T1& b) {
      using std::exp;
      if (a > b)
        return a + log1p(exp(b - a));
      return b + log1p(exp(a - b));
    }

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
    template <typename T>
    T log_sum_exp(const std::vector<T>& x) {
      using std::numeric_limits;
      using std::log;
      using std::exp;
      T max = -numeric_limits<T>::infinity();
      for (size_t ii = 0; ii < x.size(); ii++) 
        if (x[ii] > max) 
          max = x[ii];
            
      T sum = 0.0;
      for (size_t ii = 0; ii < x.size(); ii++) 
        if (x[ii] != -numeric_limits<double>::infinity()) 
          sum += exp(x[ii] - max);
          
      return max + log(sum);
    }


  }
}

#endif
