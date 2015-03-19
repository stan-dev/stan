#ifndef STAN__MATH__REV__SCAL__FUN__GAMMA_Q_HPP
#define STAN__MATH__REV__SCAL__FUN__GAMMA_Q_HPP

#include <valarray>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class gamma_q_vv_vari : public op_vv_vari {
      public:
        gamma_q_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::gamma_q(avi->val_,bvi->val_),
                     avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_
            * stan::math::grad_reg_inc_gamma(avi_->val_, bvi_->val_,
                                          boost::math::tgamma(avi_->val_),
                                          boost::math::digamma(avi_->val_));
          bvi_->adj_ -= adj_
            * boost::math::gamma_p_derivative(avi_->val_, bvi_->val_);
        }
      };

      class gamma_q_vd_vari : public op_vd_vari {
      public:
        gamma_q_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::gamma_q(avi->val_,b),
                     avi,b) {
        }
        void chain() {
          avi_->adj_ += adj_
            * stan::math::grad_reg_inc_gamma(avi_->val_, bd_,
                                          boost::math::tgamma(avi_->val_),
                                          boost::math::digamma(avi_->val_));
        }
      };

      class gamma_q_dv_vari : public op_dv_vari {
      public:
        gamma_q_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::gamma_q(a,bvi->val_),
                     a,bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_
            * boost::math::gamma_p_derivative(ad_, bvi_->val_);
        }
      };
    }

    inline var gamma_q(const stan::agrad::var& a,
                       const stan::agrad::var& b) {
      return var(new gamma_q_vv_vari(a.vi_,b.vi_));
    }

    inline var gamma_q(const stan::agrad::var& a,
                       const double& b) {
      return var(new gamma_q_vd_vari(a.vi_,b));
    }

    inline var gamma_q(const double& a,
                       const stan::agrad::var& b) {
      return var(new gamma_q_dv_vari(a,b.vi_));
    }

  }
}
#endif
