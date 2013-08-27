#ifndef __STAN__DIFF__REV__PHI_APPROX_HPP__
#define __STAN__DIFF__REV__PHI_APPROX_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/precomp_v_vari.hpp>
#include <stan/math/functions/inv_logit.hpp>

namespace stan {
  namespace diff {

    /**
     * Approximation of the unit normal CDF for variables (stan).
     *
     * http://www.jiem.org/index.php/jiem/article/download/60/27
     *
     * @param a Variable argument.
     * @return The corresponding unit normal cdf approximation.
     */
    inline var Phi_approx(const stan::diff::var& a) {
      // return inv_logit(0.07056 * pow(a,3.0) + 1.5976 * a);

      double av = a.vi_->val_;
      double av_squared = av * av;
      double av_cubed = av * av_squared;
      double f = stan::math::inv_logit(0.07056 * av_cubed + 1.5976 * av);
      double da = f * (1 - f) * (3.0 * 0.07056 * av_squared + 1.5976);
      return var(new precomp_v_vari(f,a.vi_,da));
    }

  }
}
#endif
