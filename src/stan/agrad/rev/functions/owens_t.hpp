#ifndef STAN__AGRAD__REV__FUNCTIONS__OWENS__T_HPP
#define STAN__AGRAD__REV__FUNCTIONS__OWENS__T_HPP

#include <math.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>
#include <boost/math/special_functions/owens_t.hpp>

namespace stan {
  namespace agrad {

    namespace {

      using stan::math::pi;
      class owens_t_vv_vari : public op_vv_vari {
      public:
        owens_t_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(boost::math::owens_t(avi->val_, bvi->val_), avi, bvi) {
        }
        void chain() {
          using stan::math::INV_SQRT_TWO_PI;
          using stan::math::INV_SQRT_2;
          const double neg_avi_sq_div_2 = -stan::math::square(avi_->val_) * 0.5;
          const double one_p_bvi_sq = 1.0 + stan::math::square(bvi_->val_);

          avi_->adj_ += adj_ * ::erf(bvi_->val_ * avi_->val_ * INV_SQRT_2)
            * std::exp(neg_avi_sq_div_2) * INV_SQRT_TWO_PI * -0.5;
          bvi_->adj_ += adj_ * std::exp(neg_avi_sq_div_2 * one_p_bvi_sq) 
            / (one_p_bvi_sq * 2.0 * pi());
        }
      };

      class owens_t_vd_vari : public op_vd_vari {
      public:
        owens_t_vd_vari(vari* avi, double b) :
          op_vd_vari(boost::math::owens_t(avi->val_, b), avi, b) {
        }
        void chain() {
          using stan::math::INV_SQRT_TWO_PI;
          using stan::math::INV_SQRT_2;
          using stan::math::square;
          
          avi_->adj_ += adj_ * ::erf(bd_ * avi_->val_ * INV_SQRT_2)
            * std::exp(-square(avi_->val_) * 0.5)
            * INV_SQRT_TWO_PI * -0.5;
        }
      };

      class owens_t_dv_vari : public op_dv_vari {
      public:
        owens_t_dv_vari(double a, vari* bvi) :
          op_dv_vari(boost::math::owens_t(a, bvi->val_), a, bvi) {
        }
        void chain() {
          using stan::math::INV_SQRT_2;
          using stan::math::INV_SQRT_TWO_PI;
          using stan::math::square;
          const double one_p_bvi_sq = 1.0 + stan::math::square(bvi_->val_);

          bvi_->adj_ += adj_ * std::exp(-0.5 * square(ad_)
                                        * one_p_bvi_sq)
            / (one_p_bvi_sq * 2.0 * pi());
        }
      };
    }

    /**
     * The Owen's T function of h and a.
     *
     * Used to compute the cumulative density function for the skew normal
     * distribution.
     * 
     * @param h var parameter.
     * @param a var parameter.
     * 
     * @return The Owen's T function.
     */
    inline var owens_t(const var& h, 
                       const var& a) {
      return var(new owens_t_vv_vari(h.vi_, a.vi_));
    }

    /** 
     * The Owen's T function of h and a.
     *
     * Used to compute the cumulative density function for the skew normal
     * distribution.
     * 
     * @param h var parameter.
     * @param a double parameter.
     * 
     * @return The Owen's T function.
     */
    inline var owens_t(const var& h, 
                       const double& a) {
      return var(new owens_t_vd_vari(h.vi_, a));
    }

    /** 
     * The Owen's T function of h and a.
     *
     * Used to compute the cumulative density function for the skew normal
     * distribution.
     * 
     * @param h double parameter.
     * @param a var parameter.
     * 
     * @return The Owen's T function.
     */
    inline var owens_t(const double& h, 
                       const var& a) {
      return var(new owens_t_dv_vari(h, a.vi_));
    }

  }
}
#endif
