#ifndef STAN__MATH__REV__SCAL__FUN__FMA_HPP
#define STAN__MATH__REV__SCAL__FUN__FMA_HPP

#include <cmath>
#include <valarray>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/prim/scal/meta/likely.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class fma_vvv_vari : public op_vvv_vari {
      public:
        fma_vvv_vari(vari* avi, vari* bvi, vari* cvi) :
          op_vvv_vari(::fma(avi->val_, bvi->val_, cvi->val_),
                      avi,bvi,cvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_)
                       || boost::math::isnan(cvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            cvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_ * bvi_->val_;
            bvi_->adj_ += adj_ * avi_->val_;
            cvi_->adj_ += adj_;
          }
        }
      };

      class fma_vvd_vari : public op_vvd_vari {
      public:
        fma_vvd_vari(vari* avi, vari* bvi, double c) :
          op_vvd_vari(::fma(avi->val_, bvi->val_, c),
                      avi,bvi,c) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_)
                       || boost::math::isnan(cd_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_ * bvi_->val_;
            bvi_->adj_ += adj_ * avi_->val_;
          }
        }
      };

      class fma_vdv_vari : public op_vdv_vari {
      public:
        fma_vdv_vari(vari* avi, double b, vari* cvi) :
          op_vdv_vari(::fma(avi->val_ , b, cvi->val_),
                      avi,b,cvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(cvi_->val_)
                       || boost::math::isnan(bd_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            cvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_ * bd_;
            cvi_->adj_ += adj_;
          }
        }
      };

      class fma_vdd_vari : public op_vdd_vari {
      public:
        fma_vdd_vari(vari* avi, double b, double c) :
          op_vdd_vari(::fma(avi->val_ , b, c),
                      avi,b,c) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)
                       || boost::math::isnan(cd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_ * bd_;
        }
      };

      class fma_ddv_vari : public op_ddv_vari {
      public:
        fma_ddv_vari(double a, double b, vari* cvi) :
          op_ddv_vari(::fma(a, b, cvi->val_),
                      a,b,cvi) {
        }
        void chain() {

          if (unlikely(boost::math::isnan(cvi_->val_)
                       || boost::math::isnan(ad_)
                       || boost::math::isnan(bd_)))
            cvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            cvi_->adj_ += adj_;
        }
      };
    }

    /**
     * The fused multiply-add function for three variables (C99).
     * This function returns the product of the first two arguments
     * plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
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

    /**
     * The fused multiply-add function for two variables and a value
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) + c = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) + c = x\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const stan::agrad::var& b,
                   const double& c) {
      return var(new fma_vvd_vari(a.vi_,b.vi_,c));
    }

    /**
     * The fused multiply-add function for a variable, value, and
     * variable (C99).  This function returns the product of the first
     * two arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * c) + z = c\f$, and
     *
     * \f$\frac{\partial}{\partial z} (x * c) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const double& b,
                   const stan::agrad::var& c) {
      return var(new fma_vdv_vari(a.vi_,b,c.vi_));
    }

    /**
     * The fused multiply-add function for a variable and two values
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The derivative is
     *
     * \f$\frac{d}{d x} (x * c) + d = c\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const double& b,
                   const double& c) {
      return var(new fma_vdd_vari(a.vi_,b,c));
    }

    /**
     * The fused multiply-add function for a value, variable, and
     * value (C99).  This function returns the product of the first
     * two arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The derivative is
     *
     * \f$\frac{d}{d y} (c * y) + d = c\f$, and
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const stan::agrad::var& b,
                   const double& c) {
      return var(new fma_vdd_vari(b.vi_,a,c));
    }

    /**
     * The fused multiply-add function for two values and a variable,
     * and value (C99).  This function returns the product of the
     * first two arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The derivative is
     *
     * \f$\frac{\partial}{\partial z} (c * d) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const double& b,
                   const stan::agrad::var& c) {
      return var(new fma_ddv_vari(a,b,c.vi_));
    }

    /**
     * The fused multiply-add function for a value and two variables
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * The double-based version
     * <code>::%fma(double,double,double)</code> is defined in <code>&lt;cmath&gt;</code>.
     *
     * The partial derivaties are
     *
     * \f$\frac{\partial}{\partial y} (c * y) + z = c\f$, and
     *
     * \f$\frac{\partial}{\partial z} (c * y) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const stan::agrad::var& b,
                   const stan::agrad::var& c) {
      return var(new fma_vdv_vari(b.vi_,a,c.vi_)); // a-b symmetry
    }

  }
}
#endif
