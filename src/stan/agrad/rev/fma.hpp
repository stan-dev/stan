#ifndef __STAN__AGRAD__REV__FMA_HPP__
#define __STAN__AGRAD__REV__FMA_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/vvv_vari.hpp>
#include <stan/agrad/rev/op/vvd_vari.hpp>
#include <stan/agrad/rev/op/vdv_vari.hpp>
#include <stan/agrad/rev/op/vdd_vari.hpp>
#include <stan/agrad/rev/op/ddv_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class fma_vvv_vari : public op_vvv_vari {
      public:
        fma_vvv_vari(vari* avi, vari* bvi, vari* cvi) :
          op_vvv_vari(avi->val_ * bvi->val_ + cvi->val_,
                      avi,bvi,cvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bvi_->val_;
          bvi_->adj_ += adj_ * avi_->val_;
          cvi_->adj_ += adj_;
        }
      };

      class fma_vvd_vari : public op_vv_vari {
      public:
        fma_vvd_vari(vari* avi, vari* bvi, double c) :
          op_vv_vari(avi->val_ * bvi->val_ + c,
                     avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bvi_->val_;
          bvi_->adj_ += adj_ * avi_->val_;
        }
      };

      class fma_vdv_vari : public op_vdv_vari {
      public:
        fma_vdv_vari(vari* avi, double b, vari* cvi) :
          op_vdv_vari(avi->val_ * b + cvi->val_,
                      avi,b,cvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
          cvi_->adj_ += adj_;
        }
      };

      class fma_vdd_vari : public op_vd_vari {
      public:
        fma_vdd_vari(vari* avi, double b, double c) : 
          op_vd_vari(avi->val_ * b + c,
                     avi,b) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
        }
      };

      class fma_ddv_vari : public op_v_vari {
      public:
        fma_ddv_vari(double a, double b, vari* cvi) :
          op_v_vari(a * b + cvi->val_, 
                    cvi) {
        }
        void chain() {
          // avi_ is cvi from constructor
          avi_->adj_ += adj_;
        }
      };
    }

    /**
     * The fused multiply-add function for three variables (C99).
     * This function returns the product of the first two arguments
     * plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) + z = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) + z = x\f$, and
     *
     * \f$\frac{\partial}{\partial z} (x * y) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const stan::agrad::var& b,
                   const stan::agrad::var& c) {
      return var(new fma_vvv_vari(a.vi_,b.vi_,c.vi_));
    }

  }
}
#endif
