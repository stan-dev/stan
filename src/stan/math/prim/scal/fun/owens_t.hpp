#ifndef STAN__MATH__PRIM__SCAL__FUN__OWENS_T_HPP
#define STAN__MATH__PRIM__SCAL__FUN__OWENS_T_HPP

#include <boost/math/tools/promotion.hpp>
#include <boost/math/special_functions/owens_t.hpp>

namespace stan {
  namespace math {


    /**
     * The Owen's T function of h and a.
     *
     * Used to compute the cumulative density function for the skew normal
     * distribution.
     *
     *
       \f[
       \mbox{owens\_t}(h,a) =
       \begin{cases}
         \mbox{owens\_t}(h,a) & \mbox{if } -\infty\leq h,a \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } h = \textrm{NaN or } a = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{owens\_t}(h,a)}{\partial h} =
       \begin{cases}
         \frac{\partial\, \mbox{owens\_t}(h,a)}{\partial h} & \mbox{if } -\infty\leq h,a\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } h = \textrm{NaN or } a = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{owens\_t}(h,a)}{\partial a} =
       \begin{cases}
         \frac{\partial\, \mbox{owens\_t}(h,a)}{\partial a} & \mbox{if } -\infty\leq h,a\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } h = \textrm{NaN or } a = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \mbox{owens\_t}(h,a) = \frac{1}{2\pi} \int_0^a \frac{\exp(-\frac{1}{2}h^2(1+x^2))}{1+x^2}dx
       \f]

       \f[
       \frac{\partial \, \mbox{owens\_t}(h,a)}{\partial h} = -\frac{1}{2\sqrt{2\pi}}
       \operatorname{erf}\left(\frac{ha}{\sqrt{2}}\right)
       \exp\left(-\frac{h^2}{2}\right)
       \f]

       \f[
       \frac{\partial \, \mbox{owens\_t}(h,a)}{\partial a} = \frac{\exp\left(-\frac{1}{2}h^2(1+a^2)\right)}{2\pi (1+a^2)}
       \f]
     *
     * @tparam T1 Type of first argument.
     * @tparam T2 Type of second argument.
     * @param h First argument
     * @param a Second argument
     * @return The Owen's T function.
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    owens_t(const T1& h, const T2& a) {
      return boost::math::owens_t(h, a);
    }
  }
}

#endif
