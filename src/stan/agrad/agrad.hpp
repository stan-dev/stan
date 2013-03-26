#ifndef __STAN__AGRAD__AGRAD_HPP__
#define __STAN__AGRAD__AGRAD_HPP__

#include <stan/agrad/var_stack.hpp>
#include <stan/agrad/chainable.hpp>
#include <stan/agrad/vari/vari.hpp>
#include <stan/agrad/var.hpp>
#include <stan/agrad/print_stack.hpp>

#include <stan/agrad/vari/op_v_vari.hpp>
#include <stan/agrad/vari/op_vv_vari.hpp>
#include <stan/agrad/vari/op_vd_vari.hpp>
#include <stan/agrad/vari/op_dv_vari.hpp>
#include <stan/agrad/vari/op_vvv_vari.hpp>
#include <stan/agrad/vari/op_vvd_vari.hpp>
#include <stan/agrad/vari/op_vdv_vari.hpp>
#include <stan/agrad/vari/op_vdd_vari.hpp>
#include <stan/agrad/vari/op_dvv_vari.hpp>
#include <stan/agrad/vari/op_dvd_vari.hpp>
#include <stan/agrad/vari/op_ddv_vari.hpp>
#include <stan/agrad/vari/precomp_v_vari.hpp>

#include <stan/agrad/vari/neg_vari.hpp>


#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <ostream>
#include <iostream>

#include <stan/memory/stack_alloc.hpp>

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

      class increment_vari : public op_v_vari {
      public:
        increment_vari(vari* avi) :
          op_v_vari(avi->val_ + 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class decrement_vari : public op_v_vari {
      public:
        decrement_vari(vari* avi) :
          op_v_vari(avi->val_ - 1.0, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

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

      class multiply_vv_vari : public op_vv_vari {
      public:
        multiply_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ * bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += bvi_->val_ * adj_;
          bvi_->adj_ += avi_->val_ * adj_;
        }
      };

      class multiply_vd_vari : public op_vd_vari {
      public:
        multiply_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ * b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
        }
      };

      // (a/b)' = a' * (1 / b) - b' * (a / [b * b])
      class divide_vv_vari : public op_vv_vari {
      public:
        divide_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ / bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ / bvi_->val_;
          bvi_->adj_ -= adj_ * avi_->val_ / (bvi_->val_ * bvi_->val_);
        }
      };

      class divide_vd_vari : public op_vd_vari {
      public:
        divide_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ / b, avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ / bd_;
        }
      };

      class divide_dv_vari : public op_dv_vari {
      public:
        divide_dv_vari(double a, vari* bvi) :
          op_dv_vari(a / bvi->val_, a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_ * ad_ / (bvi_->val_ * bvi_->val_);
        }
      };

      class exp_vari : public op_v_vari {
      public:
        exp_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };

      class log_vari : public op_v_vari {
      public:
        log_vari(vari* avi) :
          op_v_vari(std::log(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / avi_->val_;
        }
      };

      double LOG_10 = std::log(10.0);
    
      class log10_vari : public op_v_vari {
      public:
        const double exp_val_;
        log10_vari(vari* avi) :
          op_v_vari(std::log10(avi->val_),avi),
          exp_val_(avi->val_) {
        }
        void chain() {
          avi_->adj_ += adj_ / (LOG_10 * exp_val_);
        }
      };

      class sqrt_vari : public op_v_vari {
      public:
        sqrt_vari(vari* avi) :
          op_v_vari(std::sqrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (2.0 * val_);
        }
      };

      class pow_vv_vari : public op_vv_vari {
      public:
        pow_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::pow(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          if (avi_->val_ == 0.0) return; // partials zero, avoids 0 & log(0)
          avi_->adj_ += adj_ * bvi_->val_ * val_ / avi_->val_;
          bvi_->adj_ += adj_ * std::log(avi_->val_) * val_;
        }
      };

      class pow_vd_vari : public op_vd_vari {
      public:
        pow_vd_vari(vari* avi, double b) :
          op_vd_vari(std::pow(avi->val_,b),avi,b) {
        }
        void chain() {
          if (avi_->val_ == 0.0) return; // partials zero, avoids 0 & log(0)
          avi_->adj_ += adj_ * bd_ * val_ / avi_->val_;
        }
      };

      class pow_dv_vari : public op_dv_vari {
      public:
        pow_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::pow(a,bvi->val_),a,bvi) {
        }
        void chain() {
          if (ad_ == 0.0) return; // partials zero, avoids 0 & log(0)
          bvi_->adj_ += adj_ * std::log(ad_) * val_;
        }
      };

      class cos_vari : public op_v_vari {
      public:
        cos_vari(vari* avi) :
          op_v_vari(std::cos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * std::sin(avi_->val_);
        }
      };

      class sin_vari : public op_v_vari {
      public:
        sin_vari(vari* avi) :
          op_v_vari(std::sin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cos(avi_->val_);
        }
      };

      class tan_vari : public op_v_vari {
      public:
        tan_vari(vari* avi) :
          op_v_vari(std::tan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * (1.0 + val_ * val_); 
        }
      };

      class acos_vari : public op_v_vari {
      public:
        acos_vari(vari* avi) :
          op_v_vari(std::acos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };

      class asin_vari : public op_v_vari {
      public:
        asin_vari(vari* avi) :
          op_v_vari(std::asin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };

      class atan_vari : public op_v_vari {
      public:
        atan_vari(vari* avi) :
          op_v_vari(std::atan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 + (avi_->val_ * avi_->val_));
        }
      };

      class atan2_vv_vari : public op_vv_vari {
      public:
        atan2_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::atan2(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bvi_->val_ * bvi_->val_);
          avi_->adj_ += adj_ * bvi_->val_ / a_sq_plus_b_sq;
          bvi_->adj_ -= adj_ * avi_->val_ / a_sq_plus_b_sq;
        }
      };

      class atan2_vd_vari : public op_vd_vari {
      public:
        atan2_vd_vari(vari* avi, double b) :
          op_vd_vari(std::atan2(avi->val_,b),avi,b) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bd_ * bd_);
          avi_->adj_ += adj_ * bd_ / a_sq_plus_b_sq;
        }
      };

      class atan2_dv_vari : public op_dv_vari {
      public:
        atan2_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::atan2(a,bvi->val_),a,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (ad_ * ad_) + (bvi_->val_ * bvi_->val_);
          bvi_->adj_ -= adj_ * ad_ / a_sq_plus_b_sq;
        }
      };

      class cosh_vari : public op_v_vari {
      public:
        cosh_vari(vari* avi) :
          op_v_vari(std::cosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::sinh(avi_->val_);
        }
      };

      class sinh_vari : public op_v_vari {
      public:
        sinh_vari(vari* avi) :
          op_v_vari(std::sinh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cosh(avi_->val_);
        }
      };

      class tanh_vari : public op_v_vari {
      public:
        tanh_vari(vari* avi) :
          op_v_vari(std::tanh(avi->val_),avi) {
        }
        void chain() {
          double cosh = std::cosh(avi_->val_);
          avi_->adj_ += adj_ / (cosh * cosh);
        }
      };


      class floor_vari : public vari {
      public:
        floor_vari(vari* avi) :
          vari(std::floor(avi->val_)) {
        }
      };

      class ceil_vari : public vari {
      public:
        ceil_vari(vari* avi) :
          vari(std::ceil(avi->val_)) {
        }
      };

      class fmod_vv_vari : public op_vv_vari {
      public:
        fmod_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::fmod(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ -= adj_ * static_cast<int>(avi_->val_ / bvi_->val_);
        }
      };

      class fmod_vd_vari : public op_v_vari {
      public:
        fmod_vd_vari(vari* avi, double b) :
          op_v_vari(std::fmod(avi->val_,b),avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };
      
      class fmod_dv_vari : public op_dv_vari {
      public:
        fmod_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::fmod(a,bvi->val_),a,bvi) {
        }
        void chain() {
          int d = static_cast<int>(ad_ / bvi_->val_);
          bvi_->adj_ -= adj_ * d;
        }
      };
    }


    // COMPARISON OPERATORS

    /**
     * Equality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is the same as the
     * second's.
     */
    inline bool operator==(const var& a, const var& b) {
      return a.val() == b.val();
    }

    /**
     * Equality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is the same as the
     * second value.
     */
    inline bool operator==(const var& a, const double b) {
      return a.val() == b;
    }
  
    /**
     * Equality operator comparing a scalar and a variable's value
     * (C++).
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return True if the variable's value is equal to the scalar.
     */
    inline bool operator==(const double a, const var& b) {
      return a == b.val();
    }

    /**
     * Inequality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is not the same as the
     * second's.
     */
    inline bool operator!=(const var& a, const var& b) {
      return a.val() != b.val();
    }

    /**
     * Inequality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is not the same as the
     * second value.
     */
    inline bool operator!=(const var& a, const double b) {
      return a.val() != b;
    }

    /**
     * Inequality operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable. 
     * @return True if the first value is not the same as the
     * second variable's value.
     */
    inline bool operator!=(const double a, const var& b) {
      return a != b.val();
    }

    /**
     * Less than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than second's.
     */
    inline bool operator<(const var& a, const var& b) {
      return a.val() < b.val();
    }

    /**
     * Less than operator comparing variable's value and a double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than second value.
     */
    inline bool operator<(const var& a, const double b) {
      return a.val() < b;
    }

    /**
     * Less than operator comparing a double and variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than second variable's value.
     */
    inline bool operator<(const double a, const var& b) {
      return a < b.val();
    }

    /**
     * Greater than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than second's.
     */
    inline bool operator>(const var& a, const var& b) {
      return a.val() > b.val();
    }

    /**
     * Greater than operator comparing variable's value and double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than second value.
     */
    inline bool operator>(const var& a, const double b) {
      return a.val() > b;
    }

    /**
     * Greater than operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is greater than second variable's value.
     */
    inline bool operator>(const double a, const var& b) {
      return a > b.val();
    }

    /**
     * Less than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than or equal to
     * the second's.
     */
    inline bool operator<=(const var& a, const var& b) {
      return a.val() <= b.val();
    }

    /**
     * Less than or equal operator comparing a variable's value and a
     * scalar (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than or equal to
     * the second value.
     */
    inline bool operator<=(const var& a, const double b) {
      return a.val() <= b;
    }

    /**
     * Less than or equal operator comparing a double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than or equal to the second
     * variable's value.
     */
    inline bool operator<=(const double a, const var& b) {
      return a <= b.val();
    }

    /**
     * Greater than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than or equal
     * to the second's.
     */
    inline bool operator>=(const var& a, const var& b) {
      return a.val() >= b.val();
    }

    /**
     * Greater than or equal operator comparing variable's value and
     * double (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than or equal
     * to second value.
     */
    inline bool operator>=(const var& a, const double b) {
      return a.val() >= b;
    }

    /**
     * Greater than or equal operator comparing double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if the first value is greater than or equal to the
     * second variable's value.
     */
    inline bool operator>=(const double a, const var& b) {
      return a >= b.val();
    }

    // LOGICAL OPERATORS

    /**
     * Prefix logical negation for the value of variables (C++).  The
     * expression (!a) is equivalent to negating the scalar value of
     * the variable a.
     *
     * Note that this is the only logical operator defined for
     * variables.  Overridden logical conjunction (&&) and disjunction
     * (||) operators do not apply the same "short circuit" rules
     * as the built-in logical operators.  
     *
     * @param a Variable to negate.
     * @return True if variable is non-zero.
     */
    inline bool operator!(const var& a) {
      return !a.val();
    }


    // ARITHMETIC OPERATORS

    /**
     * Unary plus operator for variables (C++).  
     *
     * The function simply returns its input, because
     *
     * \f$\frac{d}{dx} +x = \frac{d}{dx} x = 1\f$.
     *
     * The effect of unary plus on a built-in C++ scalar type is
     * integer promotion.  Because variables are all 
     * double-precision floating point already, promotion is
     * not necessary.
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    inline var operator+(const var& a) {
      return a;
    }

    /**
     * Unary negation operator for variables (C++).
     *
     * \f$\frac{d}{dx} -x = -1\f$.
     *
     * @param a Argument variable.
     * @return Negation of variable.
     */
    inline var operator-(const var& a) {
      return var(new neg_vari(a.vi_));
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

    /**
     * Multiplication operator for two variables (C++).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) = x\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const var& b) {
      return var(new multiply_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Multiplication operator for a variable and a scalar (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial x} (x * c) = c\f$, and
     * 
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const double b) {
      if (b == 1.0)
        return a;
      return var(new multiply_vd_vari(a.vi_,b));
    }

    /**
     * Multiplication operator for a scalar and a variable (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial y} (c * y) = c\f$.
     *
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of multiplying the operands.
     */
    inline var operator*(const double a, const var& b) {
      if (a == 1.0)
        return b;
      return var(new multiply_vd_vari(b.vi_,a)); // by symmetry
    }

    /**
     * Division operator for two variables (C++).
     *
     * The partial derivatives for the variables are
     *
     * \f$\frac{\partial}{\partial x} (x/y) = 1/y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x/y) = -x / y^2\f$.
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of dividing the first variable by the
     * second.
     */
    inline var operator/(const var& a, const var& b) {
      return var(new divide_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Division operator for dividing a variable by a scalar (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{\partial}{\partial x} (x/c) = 1/c\f$.
     *
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of dividing the variable by the scalar.
     */
    inline var operator/(const var& a, const double b) {
      if (b == 1.0)
        return a;
      return var(new divide_vd_vari(a.vi_,b));
    }

    /**
     * Division operator for dividing a scalar by a variable (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} (c/y) = -c / y^2\f$.
     * 
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of dividing the scalar by the variable.
     */
    inline var operator/(const double a, const var& b) {
      return var(new divide_dv_vari(a,b.vi_));
    }

    /**
     * Prefix increment operator for variables (C++).  Following C++,
     * (++a) is defined to behave exactly as (a = a + 1.0) does,
     * but is faster and uses less memory.  In particular, the
     * result is an assignable lvalue.
     *
     * @param a Variable to increment.
     * @return Reference the result of incrementing this input variable.
     */
    inline var& operator++(var& a) {
      a.vi_ = new increment_vari(a.vi_);
      return a;
    }

    /**
     * Postfix increment operator for variables (C++).  
     *
     * Following C++, the expression <code>(a++)</code> is defined to behave like
     * the sequence of operations
     *
     * <code>var temp = a;  a = a + 1.0;  return temp;</code>
     *
     * @param a Variable to increment.
     * @return Input variable. 
     */
    inline var operator++(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new increment_vari(a.vi_);
      return temp;
    }

    /**
     * Prefix decrement operator for variables (C++).  
     *
     * Following C++, <code>(--a)</code> is defined to behave exactly as 
     *
     * <code>a = a - 1.0)</code>
     *
     * does, but is faster and uses less memory.  In particular,
     * the result is an assignable lvalue.
     *
     * @param a Variable to decrement.
     * @return Reference the result of decrementing this input variable.
     */
    inline var& operator--(var& a) {
      a.vi_ = new decrement_vari(a.vi_);
      return a;
    }

    /**
     * Postfix decrement operator for variables (C++).  
     * 
     * Following C++, the expression <code>(a--)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>var temp = a;  a = a - 1.0;  return temp;</code>
     *
     * @param a Variable to decrement.
     * @return Input variable. 
     */
    inline var operator--(var& a, int /*dummy*/) {
      var temp(a);
      a.vi_ = new decrement_vari(a.vi_);
      return temp;
    }

    // CMATH EXP AND LOG
  
    /**
     * Return the exponentiation of the specified variable (cmath).
     *
     * @param a Variable to exponentiate.
     * @return Exponentiated variable.
     */
    inline var exp(const var& a) {
      return var(new exp_vari(a.vi_));
    }

    /**
     * Return the natural log of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \log x = \frac{1}{x}\f$.
     *
     * @param a Variable whose log is taken.
     * @return Natural log of variable.
     */
    inline var log(const var& a) {
      return var(new log_vari(a.vi_));
    }

    /**
     * Return the base 10 log of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \log_{10} x = \frac{1}{x \log 10}\f$.
     * 
     * @param a Variable whose log is taken.
     * @return Base 10 log of variable.
     */
    inline var log10(const var& a) {
      return var(new log10_vari(a.vi_));
    }


    // POWER FUNCTIONS

    /**
     * Return the square root of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sqrt{x} = \frac{1}{2 \sqrt{x}}\f$.
     * 
     * @param a Variable whose square root is taken.
     * @return Square root of variable.
     */
    inline var sqrt(const var& a) {
      return var(new sqrt_vari(a.vi_));
    }

    /**
     * Return the base raised to the power of the exponent (cmath).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} \mbox{pow}(x,y) = y x^{y-1}\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{pow}(x,y) = x^y \ \log x\f$.
     *
     * @param base Base variable.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const var& exponent) {
      return var(new pow_vv_vari(base.vi_,exponent.vi_));
    }
  
    /**
     * Return the base variable raised to the power of the exponent
     * scalar (cmath).
     *
     * The derivative for the variable is
     *
     * \f$\frac{d}{dx} \mbox{pow}(x,c) = c x^{c-1}\f$.
     *
     * @param base Base variable.
     * @param exponent Exponent scalar.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const double exponent) {
      if (exponent == 0.5)
        return sqrt(base);
      if (exponent == 1.0)
        return base;
      if (exponent == 2.0)
        return base * base; // FIXME: square() functionality from special_functions
      return var(new pow_vd_vari(base.vi_,exponent));
    }

    /**
     * Return the base scalar raised to the power of the exponent
     * variable (cmath).
     *
     * The derivative for the variable is
     * 
     * \f$\frac{d}{d y} \mbox{pow}(c,y) = c^y \log c \f$.
     * 
     * @param base Base scalar.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const double base, const var& exponent) {
      return var(new pow_dv_vari(base,exponent.vi_));
    }


    // TRIG FUNCTIONS

    /**
     * Return the cosine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cos x = - \sin x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Cosine of variable. 
     */
    inline var cos(const var& a) {
      return var(new cos_vari(a.vi_));
    }

    /**
     * Return the sine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sin x = \cos x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Sine of variable. 
     */
    inline var sin(const var& a) {
      return var(new sin_vari(a.vi_));
    }

    /**
     * Return the tangent of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tan x = \sec^2 x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Tangent of variable. 
     */
    inline var tan(const var& a) {
      return var(new tan_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc cosine of a variable,
     * in radians (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arccos x = \frac{-1}{\sqrt{1 - x^2}}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc cosine of variable, in radians. 
     */
    inline var acos(const var& a) {
      return var(new acos_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc sine, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arcsin x = \frac{1}{\sqrt{1 - x^2}}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc sine of variable, in radians. 
     */
    inline var asin(const var& a) {
      return var(new asin_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc tangent of variable, in radians. 
     */
    inline var atan(const var& a) {
      return var(new atan_vari(a.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second (cmath).
     *
     * The partial derivatives are defined by
     *
     * $\f$ \frac{\partial}{\partial x} \arctan \frac{x}{y} = \frac{y}{x^2 + y^2}\f$, and
     * 
     * $\f$ \frac{\partial}{\partial y} \arctan \frac{x}{y} = \frac{-x}{x^2 + y^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const var& b) {
      return var(new atan2_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * $\f$ \frac{d}{d x} \arctan \frac{x}{c} = \frac{c}{x^2 + c^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator scalar.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const double b) {
      return var(new atan2_vd_vari(a.vi_,b));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first scalar divided by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * $\f$ \frac{\partial}{\partial y} \arctan \frac{c}{y} = \frac{-c}{c^2 + y^2}\f$.
     *
     * @param a Numerator scalar.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const double a, const var& b) {
      return var(new atan2_dv_vari(a,b.vi_));
    }

    // HYPERBOLIC FUNCTIONS

    /**
     * Return the hyperbolic cosine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cosh x = \sinh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic cosine of variable.
     */
    inline var cosh(const var& a) {
      return var(new cosh_vari(a.vi_));
    }

    /**
     * Return the hyperbolic sine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sinh x = \cosh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic sine of variable.
     */
    inline var sinh(const var& a) {
      return var(new sinh_vari(a.vi_));
    }

    /**
     * Return the hyperbolic tangent of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tanh x = \frac{1}{\cosh^2 x}\f$.
     *
     * @param a Variable.
     * @return Hyperbolic tangent of variable.
     */
    inline var tanh(const var& a) {
      return var(new tanh_vari(a.vi_));
    }
  

    // ROUNDING FUNCTIONS

    /**
     * Return the absolute value of the variable (cmath).  
     *
     * Choosing an arbitrary value at the non-differentiable point 0,
     * 
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * where \f$\mbox{sgn}(x)\f$ is the signum function, taking values
     * -1 if \f$x < 0\f$, 0 if \f$x == 0\f$, and 1 if \f$x == 1\f$.
     *
     * The function <code>abs()</code> provides the same behavior, with
     * <code>abs()</code> defined in stdlib.h and <code>fabs()</code> defined in <code>cmath</code>.
     *
     * @param a Input variable.
     * @return Absolute value of variable.
     */
    inline var fabs(const var& a) {
      // cut-and-paste from abs()
      if (a.val() > 0.0)
        return a;
      if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      // FIXME:  is this right?  breaks connection to a
      return var(new vari(0.0));
    }

    /**
     * Return the floor of the specified variable (cmath).  
     *
     * The derivative of the floor function is defined and
     * zero everywhere but at integers, so we set these derivatives
     * to zero for convenience, 
     *
     * \f$\frac{d}{dx} {\lfloor x \rfloor} = 0\f$.
     *
     * The floor function rounds down.  For double values, this is the largest
     * integral value that is not greater than the specified value.
     * Although this function is not differentiable because it is
     * discontinuous at integral values, its gradient is returned as
     * zero everywhere.
     * 
     * @param a Input variable.
     * @return Floor of the variable.
     */
    inline var floor(const var& a) {
      return var(new floor_vari(a.vi_));
    }

    /**
     * Return the ceiling of the specified variable (cmath).
     *
     * The derivative of the ceiling function is defined and
     * zero everywhere but at integers, and we set them to zero for
     * convenience, 
     *
     * \f$\frac{d}{dx} {\lceil x \rceil} = 0\f$.
     *
     * The ceiling function rounds up.  For double values, this is the
     * smallest integral value that is not less than the specified
     * value.  Although this function is not differentiable because it
     * is discontinuous at integral values, its gradient is returned
     * as zero everywhere.
     * 
     * @param a Input variable.
     * @return Ceiling of the variable.
     */
    inline var ceil(const var& a) {
      return var(new ceil_vari(a.vi_));
    }

    /**
     * Return the floating point remainder after dividing the
     * first variable by the second (cmath).
     *
     * The partial derivatives with respect to the variables are defined
     * everywhere but where \f$x = y\f$, but we set these to match other values,
     * with
     *
     * \f$\frac{\partial}{\partial x} \mbox{fmod}(x,y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{fmod}(x,y) = -\lfloor \frac{x}{y} \rfloor\f$.
     *
     * @param a First variable.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing the first variable
     * by the second.
     */
    inline var fmod(const var& a, const var& b) {
      return var(new fmod_vv_vari(a.vi_,b.vi_));
    }
  
    /**
     * Return the floating point remainder after dividing the
     * the first variable by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d x} \mbox{fmod}(x,c) = \frac{1}{c}\f$.
     *
     * @param a First variable.
     * @param b Second scalar.
     * @return Floating pointer remainder of dividing the first variable by
     * the second scalar.
     */
    inline var fmod(const var& a, const double b) {
      return var(new fmod_vd_vari(a.vi_,b));
    }

    /**
     * Return the floating point remainder after dividing the
     * first scalar by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} \mbox{fmod}(c,y) = -\lfloor \frac{c}{y} \rfloor\f$.
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing first scalar by
     * the second variable.
     */
    inline var fmod(const double a, const var& b) {
      return var(new fmod_dv_vari(a,b.vi_));
    }


    // STD LIB FUNCTIONS

    /**
     * Return the absolute value of the variable (std).  
     *
     * The value at the undifferentiable point 0 is conveniently set
     * 0, so that
     *
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * The function fabs() provides identical behavior, with
     * abs() defined in stdlib.h and fabs() defined in cmath.
     *
     * @param a Variable input.
     * @return Absolute value of variable.
     */
    inline var abs(const var& a) {   
      // cut-and-paste from fabs()
      if (a.val() > 0.0)
        return a;
      if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      // FIXME:  same as fabs() -- is this right?
      return var(new vari(0.0));
    }


    /**
     * Recover memory used for all variables for reuse.
     */
    static void recover_memory() {
      var_stack_.clear();
      var_nochain_stack_.clear();
      memalloc_.recover_all();
    }

    /**
     * Return all memory used for gradients back to the system.
     */
    static void free_memory() {
      memalloc_.free_all();
    }

    /**
     * Compute the gradient for all variables starting from the
     * specified root variable implementation.  Does not recover
     * memory.  This chainable variable's adjoint is initialized
     * using the method <code>init_dependent()</code> and then the
     * chain rule is applied working down the stack from this
     * chainable and calling each chainable's <code>chain()</code>
     * method in turn.
     *
     * @param vi Variable implementation for root of partial
     * derivative propagation.
     */
    static void grad(chainable* vi) {
      std::vector<chainable*>::reverse_iterator it;

      vi->init_dependent(); 
      // propagate derivates for vars
      for (it = var_stack_.rbegin(); it < var_stack_.rend(); ++it)
        (*it)->chain();
    }

    /**
     * Reset all adjoint values in the stack to zero.
     */
    static void set_zero_all_adjoints() {
      for (size_t i = 0; i < var_stack_.size(); ++i)
        var_stack_[i]->set_zero_adjoint();
      for (size_t i = 0; i < var_nochain_stack_.size(); ++i)
        var_nochain_stack_[i]->set_zero_adjoint();
    }

    /**
     * Return the Jacobian of the function producing the specified
     * dependent variables with respect to the specified independent
     * variables. 
     *
     * A typical use case would be to take the Jacobian of a function
     * from independent variables to dependentant variables.  For instance,
     * 
     * <pre>
     * std::vector<var> f(std::vector<var>& x) { ... }
     * std::vector<var> x = ...;
     * std::vector<var> y = f(x);
     * std::vector<std::vector<double> > J;
     * jacobian(y,x,J);
     * </pre>
     *
     * After executing this code, <code>J</code> will contain the
     * Jacobian, stored as a standard vector of gradients.
     * Specifically, <code>J[m]</code> will be the gradient of <code>y[m]</code>
     * with respect to <code>x</code>, and thus <code>J[m][n]</code> will be 
     * <code><i>d</i>y[m]/<i>d</i>x[n]</code>.
     *
     * @param[in] dependents Dependent (output) variables.
     * @param[in] independents Indepent (input) variables.
     * @param[out] jacobian Jacobian of the transform.
     */
    inline void jacobian(std::vector<var>& dependents,
                         std::vector<var>& independents,
                         std::vector<std::vector<double> >& jacobian) {
      jacobian.resize(dependents.size());
      for (size_t i = 0; i < dependents.size(); ++i) {
        jacobian[i].resize(independents.size());
        if (i > 0) 
          set_zero_all_adjoints();
        jacobian.push_back(std::vector<double>(0));
        grad(dependents[i].vi_);
        for (size_t j = 0; j < independents.size(); ++j)
          jacobian[i][j] = independents[j].adj();
      }
    }


    inline var& var::operator+=(const var& b) {
      vi_ = new add_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator+=(const double b) {
      if (b == 0.0)
        return *this;
      vi_ = new add_vd_vari(vi_,b);
      return *this;
    }

    inline var& var::operator-=(const var& b) {
      vi_ = new subtract_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator-=(const double b) {
      if (b == 0.0)
        return *this;
      vi_ = new subtract_vd_vari(vi_,b);
      return *this;
    }

    inline var& var::operator*=(const var& b) {
      vi_ = new multiply_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator*=(const double b) {
      if (b == 1.0)
        return *this;
      vi_ = new multiply_vd_vari(vi_,b);
      return *this;
    }

    inline var& var::operator/=(const var& b) {
        vi_ = new divide_vv_vari(vi_,b.vi_);
        return *this;
      }

    inline var& var::operator/=(const double b) {
        if (b == 1.0)
          return *this;
        vi_ = new divide_vd_vari(vi_,b);
        return *this;
      }

    template <typename T>
    inline bool is_uninitialized(T x) {
      return false;
    }
    inline bool is_uninitialized(var x) {
      return x.is_uninitialized();
    }


  }
}


namespace std {

  /** 
   * Specialization of numeric limits for var objects.
   *
   * This implementation of std::numeric_limits<stan::agrad::var>
   * is used to treat var objects like doubles.
   */
  template<> 
  struct numeric_limits<stan::agrad::var> {
    static const bool is_specialized = true;
    static stan::agrad::var min() { return numeric_limits<double>::min(); }
    static stan::agrad::var max() { return numeric_limits<double>::max(); }
    static const int digits = numeric_limits<double>::digits;
    static const int digits10 = numeric_limits<double>::digits10;
    static const bool is_signed = numeric_limits<double>::is_signed;
    static const bool is_integer = numeric_limits<double>::is_integer;
    static const bool is_exact = numeric_limits<double>::is_exact;
    static const int radix = numeric_limits<double>::radix;
    static stan::agrad::var epsilon() { return numeric_limits<double>::epsilon(); }
    static stan::agrad::var round_error() { return numeric_limits<double>::round_error(); }

    static const int  min_exponent = numeric_limits<double>::min_exponent;
    static const int  min_exponent10 = numeric_limits<double>::min_exponent10;
    static const int  max_exponent = numeric_limits<double>::max_exponent;
    static const int  max_exponent10 = numeric_limits<double>::max_exponent10;

    static const bool has_infinity = numeric_limits<double>::has_infinity;
    static const bool has_quiet_NaN = numeric_limits<double>::has_quiet_NaN;
    static const bool has_signaling_NaN = numeric_limits<double>::has_signaling_NaN;
    static const float_denorm_style has_denorm = numeric_limits<double>::has_denorm;
    static const bool has_denorm_loss = numeric_limits<double>::has_denorm_loss;
    static stan::agrad::var infinity() { return numeric_limits<double>::infinity(); }
    static stan::agrad::var quiet_NaN() { return numeric_limits<double>::quiet_NaN(); }
    static stan::agrad::var signaling_NaN() { return numeric_limits<double>::signaling_NaN(); }
    static stan::agrad::var denorm_min() { return numeric_limits<double>::denorm_min(); }

    static const bool is_iec559 = numeric_limits<double>::is_iec559;
    static const bool is_bounded = numeric_limits<double>::is_bounded;
    static const bool is_modulo = numeric_limits<double>::is_modulo;

    static const bool traps = numeric_limits<double>::traps;
    static const bool tinyness_before = numeric_limits<double>::tinyness_before;
    static const float_round_style round_style = numeric_limits<double>::round_style;
  };

  /**
   * Checks if the given number is NaN.
   * 
   * Return <code>true</code> if the value of the
   * specified variable is not a number.
   *
   * @param a Variable to test.
   * @return <code>true</code> if value is not a number.
   */
  inline int isnan(const stan::agrad::var& a) {
    return isnan(a.val());
  }

  /**
   * Checks if the given number is infinite.
   * 
   * Return <code>true</code> if the value of the
   * a is positive or negative infinity.
   *
   * @param a Variable to test.
   * @return <code>true</code> if value is infinite.
   */
  inline int isinf(const stan::agrad::var& a) {
    return isinf(a.val());
  }

}

#endif
