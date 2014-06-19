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
#include <stan/math/constants.hpp>

#include <stan/prob/internal_math/math/grad_inc_beta.hpp>
#include <stan/prob/internal_math/math/inc_beta.hpp>

#include <stan/agrad/fwd/functions/pow.hpp>
#include <stan/agrad/rev/functions/pow.hpp>

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
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,avi_->val_,bvi_->val_,cvi_->val_);

          avi_->adj_ += adj_ * d_a;
          bvi_->adj_ += adj_ * d_b;
          cvi_->adj_ += adj_ * std::pow((1-cvi_->val_),bvi_->val_-1)
            * std::pow(cvi_->val_,avi_->val_-1);
        }
      };

      class inc_beta_vvd_vari : public op_vvd_vari {
      public:
        inc_beta_vvd_vari(vari* avi, vari* bvi, double c) :
          op_vvd_vari(stan::math::inc_beta(avi->val_, bvi->val_, c),
                      avi,bvi,c) {
        }
        void chain() {
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,avi_->val_,bvi_->val_,cd_);

          avi_->adj_ += adj_ * d_a;
          bvi_->adj_ += adj_ * d_b;
        }
      };
      class inc_beta_vdv_vari : public op_vdv_vari {
      public:
        inc_beta_vdv_vari(vari* avi, double b, vari* cvi) :
          op_vdv_vari(stan::math::inc_beta(avi->val_, b, cvi->val_),
                      avi,b,cvi) {
        }
        void chain() {
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,avi_->val_,bd_,cvi_->val_);

          avi_->adj_ += adj_ * d_a;
          cvi_->adj_ += adj_ * std::pow((1-cvi_->val_),bd_-1)
            * std::pow(cvi_->val_,avi_->val_-1);
        }
      };
      class inc_beta_dvv_vari : public op_dvv_vari {
      public:
        inc_beta_dvv_vari(double a, vari* bvi, vari* cvi) :
          op_dvv_vari(stan::math::inc_beta(a, bvi->val_, cvi->val_),
                      a,bvi,cvi) {
        }
        void chain() {
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,ad_,bvi_->val_,cvi_->val_);

          bvi_->adj_ += adj_ * d_b;
          cvi_->adj_ += adj_ * std::pow((1-cvi_->val_),bvi_->val_-1)
            * std::pow(cvi_->val_,ad_-1);
        }
      };
      class inc_beta_vdd_vari : public op_vdd_vari {
      public:
        inc_beta_vdd_vari(vari* avi, double b, double c) :
          op_vdd_vari(stan::math::inc_beta(avi->val_, b, c),
                      avi,b,c) {
        }
        void chain() {
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,avi_->val_,bd_,cd_);

          avi_->adj_ += adj_ * d_a;
        }
      };

      class inc_beta_dvd_vari : public op_dvd_vari {
      public:
        inc_beta_dvd_vari(double a, vari* bvi, double c) :
          op_dvd_vari(stan::math::inc_beta(a, bvi->val_, c),
                      a,bvi,c) {
        }
        void chain() {
          double d_a; double d_b;
          stan::math::grad_inc_beta(d_a,d_b,ad_,bvi_->val_,cd_);

          bvi_->adj_ += adj_ * d_b;
        }
      };
      class inc_beta_ddv_vari : public op_ddv_vari {
      public:
        inc_beta_ddv_vari(double a, double b, vari* cvi) :
          op_ddv_vari(stan::math::inc_beta(a,b, cvi->val_),
                      a,b,cvi) {
        }
        void chain() {
          cvi_->adj_ += adj_ * std::pow((1-cvi_->val_),bd_-1)
            * std::pow(cvi_->val_,ad_-1);
        }
      };

    }

    inline var inc_beta(const stan::agrad::var& a,
                        const stan::agrad::var& b,
                        const stan::agrad::var& c) {
      return var(new inc_beta_vvv_vari(a.vi_,b.vi_,c.vi_));
    }
    inline var inc_beta(const stan::agrad::var& a,
                        const stan::agrad::var& b,
                        const double& c) {
      return var(new inc_beta_vvd_vari(a.vi_,b.vi_,c));
    }

    inline var inc_beta(const stan::agrad::var& a,
                        const double& b,
                        const stan::agrad::var& c) {
      return var(new inc_beta_vdv_vari(a.vi_,b,c.vi_));
    }
    
    inline var inc_beta(const double& a,
                        const stan::agrad::var& b,
                        const stan::agrad::var& c) {
      return var(new inc_beta_dvv_vari(a,b.vi_,c.vi_));
    }


    inline var inc_beta(const stan::agrad::var& a,
                        const double& b,
                        const double& c) {
      return var(new inc_beta_vdd_vari(a.vi_,b,c));
    }

    inline var inc_beta(const double& a,
                        const stan::agrad::var& b,
                        const double& c) {
      return var(new inc_beta_dvd_vari(a,b.vi_,c));
    }

    inline var inc_beta(const double& a,
                        const double& b,
                        const stan::agrad::var& c) {
      return var(new inc_beta_ddv_vari(a,b,c.vi_));
    }
  }
}
#endif
