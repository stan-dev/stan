#ifndef __STAN__AGRAD__AGRAD_HPP__
#define __STAN__AGRAD__AGRAD_HPP__

#include <stan/agrad/rev/var_stack.hpp>
#include <stan/agrad/rev/chainable.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/print_stack.hpp>

#include <stan/agrad/rev/op/v_vari.hpp>
#include <stan/agrad/rev/op/vv_vari.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/agrad/rev/op/dv_vari.hpp>
#include <stan/agrad/rev/op/vvv_vari.hpp>
#include <stan/agrad/rev/op/vvd_vari.hpp>
#include <stan/agrad/rev/op/vdv_vari.hpp>
#include <stan/agrad/rev/op/vdd_vari.hpp>
#include <stan/agrad/rev/op/dvv_vari.hpp>
#include <stan/agrad/rev/op/dvd_vari.hpp>
#include <stan/agrad/rev/op/ddv_vari.hpp>
#include <stan/agrad/rev/precomp_v_vari.hpp>

#include <stan/agrad/rev/operator_unary_negative.hpp>
#include <stan/agrad/rev/operator_equal.hpp>
#include <stan/agrad/rev/operator_not_equal.hpp>
#include <stan/agrad/rev/operator_greater_than.hpp>
#include <stan/agrad/rev/operator_greater_than_or_equal.hpp>
#include <stan/agrad/rev/operator_less_than.hpp>
#include <stan/agrad/rev/operator_less_than_or_equal.hpp>
#include <stan/agrad/rev/operator_unary_not.hpp>
#include <stan/agrad/rev/operator_unary_plus.hpp>
#include <stan/agrad/rev/operator_addition.hpp>
#include <stan/agrad/rev/operator_subtraction.hpp>
#include <stan/agrad/rev/operator_multiplication.hpp>
#include <stan/agrad/rev/operator_division.hpp>
#include <stan/agrad/rev/operator_unary_increment.hpp>
#include <stan/agrad/rev/operator_unary_decrement.hpp>

#include <stan/agrad/rev/exp.hpp>
#include <stan/agrad/rev/log.hpp>
#include <stan/agrad/rev/log10.hpp>
#include <stan/agrad/rev/sqrt.hpp>
#include <stan/agrad/rev/pow.hpp>
#include <stan/agrad/rev/cos.hpp>
#include <stan/agrad/rev/sin.hpp>
#include <stan/agrad/rev/tan.hpp>
#include <stan/agrad/rev/acos.hpp>
#include <stan/agrad/rev/asin.hpp>
#include <stan/agrad/rev/atan.hpp>
#include <stan/agrad/rev/atan2.hpp>
#include <stan/agrad/rev/cosh.hpp>
#include <stan/agrad/rev/sinh.hpp>
#include <stan/agrad/rev/tanh.hpp>
#include <stan/agrad/rev/fabs.hpp>
#include <stan/agrad/rev/floor.hpp>
#include <stan/agrad/rev/ceil.hpp>
#include <stan/agrad/rev/fmod.hpp>
#include <stan/agrad/rev/abs.hpp>
#include <stan/agrad/rev/jacobian.hpp>

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


    // ROUNDING FUNCTIONS


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
