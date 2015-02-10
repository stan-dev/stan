#ifndef __STAN__PROB__INTERNAL_MATH__REV__INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__REV__INC_BETA_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vvv_vari.hpp>
#include <stan/agrad/rev/internal/vvd_vari.hpp>
#include <stan/agrad/rev/internal/vdv_vari.hpp>
#include <stan/agrad/rev/internal/dvv_vari.hpp>
#include <stan/agrad/rev/internal/vdd_vari.hpp>
#include <stan/agrad/rev/internal/dvd_vari.hpp>
#include <stan/agrad/rev/internal/ddv_vari.hpp>
#include <stan/math/functions/constants.hpp>

#include <stan/prob/internal_math/math/grad_reg_inc_beta.hpp>

#include <stan/agrad/rev/functions/pow.hpp>
#include <stan/math/functions/lbeta.hpp>
#include <stan/math/functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class inc_beta_vvv_vari : public op_vvv_vari {
      public:
        inc_beta_vvv_vari(vari* avi, vari* bvi, vari* cvi) :
          op_vvv_vari(stan::math::inc_beta(avi->val_, bvi->val_, cvi->val_),
                      avi,bvi,cvi) {
        }
        void chain() {
          using stan::math::digamma;
          using stan::math::lbeta;

          double d_a; double d_b;
          stan::math::grad_reg_inc_beta(d_a,d_b,avi_->val_,bvi_->val_,
                                        cvi_->val_,digamma(avi_->val_),
                                        digamma(bvi_->val_),
                                        digamma(avi_->val_ + bvi_->val_),
                                        std::exp(lbeta(avi_->val_, bvi_->val_)));

          avi_->adj_ += adj_ * d_a;
          bvi_->adj_ += adj_ * d_b;
          cvi_->adj_ += adj_ * std::pow((1-cvi_->val_),bvi_->val_-1)
            * std::pow(cvi_->val_,avi_->val_-1)
            / std::exp(stan::math::lbeta(avi_->val_,bvi_->val_));
        }
      };

    }

    inline var inc_beta(const stan::agrad::var& a,
                        const stan::agrad::var& b,
                        const stan::agrad::var& c) {
      return var(new inc_beta_vvv_vari(a.vi_,b.vi_,c.vi_));
    }

  }
}
#endif
