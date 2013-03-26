#ifndef __STAN__AGRAD__AGRAD_HPP__
#define __STAN__AGRAD__AGRAD_HPP__

#include <stan/agrad/rev/var_stack.hpp>
#include <stan/agrad/rev/set_zero_all_adjoints.hpp>
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
