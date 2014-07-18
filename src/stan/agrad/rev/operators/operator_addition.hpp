#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_ADDITION_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_ADDITION_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class add_vv_vari : public op_vv_vari {
      public:
        add_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ + bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ += adj_;
        }
      };

      class add_vd_vari : public op_vd_vari {
      public:
        add_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ + b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };
    }

    /**
     * Addition operator for variables (C++).
     *
     * The partial derivatives are defined by 
     *
     * \f$\frac{\partial}{\partial x} (x+y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x+y) = 1\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of adding two variables.
     */
    inline var operator+(const var& a, const var& b) {    
      return var(new add_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Addition operator for variable and scalar (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{dx} (x + c) = 1\f$.
     *
     * @param a First variable operand.
     * @param b Second scalar operand.
     * @return Result of adding variable and scalar.
     */
    inline var operator+(const var& a, const double b) {
      if (b == 0.0)
        return a;
      return var(new add_vd_vari(a.vi_,b));
    }

    /**
     * Addition operator for scalar and variable (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{dy} (c + y) = 1\f$.
     *
     * @param a First scalar operand.
     * @param b Second variable operand.
     * @return Result of adding variable and scalar.
     */
    inline var operator+(const double a, const var& b) {
      if (a == 0.0)
        return b;
      return var(new add_vd_vari(b.vi_,a)); // by symmetry
    }

  }
}
#endif
