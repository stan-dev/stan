#ifndef STAN__MATH__PRIM__SCAL__FUN__LOG_DIFF_EXP_HPP
#define STAN__MATH__PRIM__SCAL__FUN__LOG_DIFF_EXP_HPP

#include <boost/math/tools/promotion.hpp>
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>

namespace stan {
  namespace math {

    /**
     * The natural logarithm of the difference of the natural exponentiation
     * of x1 and the natural exponentiation of x2
     *
     * This function is only defined for x<0
     *
     *
       \f[
       \mbox{log\_diff\_exp}(x,y) =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x \leq y\\
         \ln(\exp(x)-\exp(y)) & \mbox{if } x > y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{log\_diff\_exp}(x,y)}{\partial x} =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x \leq y\\
         \frac{\exp(x)}{\exp(x)-\exp(y)} & \mbox{if } x > y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{log\_diff\_exp}(x,y)}{\partial y} =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x \leq y\\
         -\frac{\exp(y)}{\exp(x)-\exp(y)} & \mbox{if } x > y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    log_diff_exp(const T1 x, const T2 y) {
      if (x <= y)
        return std::numeric_limits<double>::quiet_NaN();
      return x + log1m_exp(y - x);
    }

  }
}

#endif
