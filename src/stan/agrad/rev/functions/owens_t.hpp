#ifndef STAN__AGRAD__REV__FUNCTIONS__OWENS__T_HPP
#define STAN__AGRAD__REV__FUNCTIONS__OWENS__T_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/math/constants.hpp>
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
          avi_->adj_ += adj_ * boost::math::erf(bvi_->val_ * avi_->val_ / std::sqrt(2.0)) * std::exp(-avi_->val_ * avi_->val_ / 2.0) * std::sqrt(pi() / 2.0) / (-2.0 * pi());
          bvi_->adj_ += adj_ * std::exp(-0.5 * avi_->val_ * avi_->val_ * (1.0 + bvi_->val_ * bvi_->val_)) / ((1 + bvi_->val_ * bvi_->val_) * 2.0 * pi());
        }
      };

      class owens_t_vd_vari : public op_vd_vari {
      public:
        owens_t_vd_vari(vari* avi, double b) :
          op_vd_vari(boost::math::owens_t(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * boost::math::erf(bd_ * avi_->val_ / std::sqrt(2.0)) * std::exp(-avi_->val_ * avi_->val_ / 2.0) * std::sqrt(pi() / 2.0) / (-2.0 * pi());
        }
      };

      class owens_t_dv_vari : public op_dv_vari {
      public:
        owens_t_dv_vari(double a, vari* bvi) :
          op_dv_vari(boost::math::owens_t(a, bvi->val_), a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * std::exp(-0.5 * ad_ * ad_ * (1.0 + bvi_->val_ * bvi_->val_)) / ((1 + bvi_->val_ * bvi_->val_) * 2.0 * pi());
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
