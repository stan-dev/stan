#ifndef __STAN__DIFF__REV__HYPOT_HPP__
#define __STAN__DIFF__REV__HYPOT_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/vv_vari.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <boost/math/special_functions/hypot.hpp>

namespace stan {
  namespace diff {

    namespace {
      class hypot_vv_vari : public op_vv_vari {
      public:
        hypot_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(boost::math::hypot(avi->val_,bvi->val_),
                     avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * avi_->val_ / val_;
          bvi_->adj_ += adj_ * bvi_->val_ / val_;
        }
      };

      class hypot_vd_vari : public op_v_vari {
      public:
        hypot_vd_vari(vari* avi, double b) :
          op_v_vari(boost::math::hypot(avi->val_,b),
                    avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * avi_->val_ / val_;
        }
      };
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The partial derivatives are given by
     *
     * \f$\frac{\partial}{\partial x} \sqrt{x^2 + y^2} = \frac{x}{\sqrt{x^2 + y^2}}\f$, and
     *
     * \f$\frac{\partial}{\partial y} \sqrt{x^2 + y^2} = \frac{y}{\sqrt{x^2 + y^2}}\f$.
     *
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const stan::diff::var& a,
                     const stan::diff::var& b) {
      return var(new hypot_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The derivative is
     *
     * \f$\frac{d}{d x} \sqrt{x^2 + c^2} = \frac{x}{\sqrt{x^2 + c^2}}\f$.
     * 
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const stan::diff::var& a,
                     const double& b) {
      return var(new hypot_vd_vari(a.vi_,b));
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The derivative is
     *
     * \f$\frac{d}{d y} \sqrt{c^2 + y^2} = \frac{y}{\sqrt{c^2 + y^2}}\f$.
     *
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const double& a,
                     const stan::diff::var& b) {
      return var(new hypot_vd_vari(b.vi_,a));
    }

  }
}
#endif
