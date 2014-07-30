#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_SUBTRACTION_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_SUBTRACTION_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class subtract_vv_vari : public op_vv_vari {
      public:
        subtract_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ - bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ -= adj_;
        }
      };
    
      class subtract_vd_vari : public op_vd_vari {
      public:
        subtract_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ - b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class subtract_dv_vari : public op_dv_vari {
      public:
        subtract_dv_vari(double a, vari* bvi) :
          op_dv_vari(a - bvi->val_, a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_;
        }
      };
    }

    /**
     * Subtraction operator for variables (C++).
     *
     * The partial derivatives are defined by 
     *
     * \f$\frac{\partial}{\partial x} (x-y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x-y) = -1\f$.
     * 
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of subtracting the second variable from
     * the first.
     */
    inline var operator-(const var& a, const var& b) {
      return var(new subtract_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Subtraction operator for variable and scalar (C++).
     *
     * The derivative for the variable is
     *
     * \f$\frac{\partial}{\partial x} (x-c) = 1\f$, and
     *
     * @param a First variable operand.
     * @param b Second scalar operand.
     * @return Result of subtracting the scalar from the variable.
     */
    inline var operator-(const var& a, const double b) {
      if (b == 0.0)
        return a;
      return var(new subtract_vd_vari(a.vi_,b));
    }

    /**
     * Subtraction operator for scalar and variable (C++).
     *
     * The derivative for the variable is
     *
     * \f$\frac{\partial}{\partial y} (c-y) = -1\f$, and
     *
     * @param a First scalar operand.
     * @param b Second variable operand.
     * @return Result of sutracting a variable from a scalar.
     */
    inline var operator-(const double a, const var& b) {
      return var(new subtract_dv_vari(a,b.vi_));
    }

  }
}
#endif
