#ifndef __STAN__AGRAD__FWD__NUMERIC__LIMITS__HPP__
#define __STAN__AGRAD__FWD__NUMERIC__LIMITS__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace std {

  template <typename T>

  struct numeric_limits<stan::agrad::fvar<T> > {
    static const bool is_specialized = true;
    static stan::agrad::fvar<T> min() { return numeric_limits<T>::min(); }
    static stan::agrad::fvar<T> max() { return numeric_limits<T>::max(); }
    static const int digits = numeric_limits<T>::digits;
    static const int digits10 = numeric_limits<T>::digits10;
    static const bool is_signed = numeric_limits<T>::is_signed;
    static const bool is_integer = numeric_limits<T>::is_integer;
    static const bool is_exact = numeric_limits<T>::is_exact;
    static const int radix = numeric_limits<T>::radix;
    static stan::agrad::fvar<T> epsilon() { 
      return numeric_limits<T>::epsilon(); }
    static stan::agrad::fvar<T> round_error() { 
      return numeric_limits<T>::round_error(); }

    static const int  min_exponent = numeric_limits<T>::min_exponent;
    static const int  min_exponent10 = numeric_limits<T>::min_exponent10;
    static const int  max_exponent = numeric_limits<T>::max_exponent;
    static const int  max_exponent10 = numeric_limits<T>::max_exponent10;

    static const bool has_infinity = numeric_limits<T>::has_infinity;
    static const bool has_quiet_NaN = numeric_limits<T>::has_quiet_NaN;
    static const bool has_signaling_NaN = numeric_limits<T>::has_signaling_NaN;
    static const float_denorm_style has_denorm = numeric_limits<T>::has_denorm;
    static const bool has_denorm_loss = numeric_limits<T>::has_denorm_loss;
    static stan::agrad::fvar<T> infinity() { 
      return numeric_limits<T>::infinity(); }
    static stan::agrad::fvar<T> quiet_NaN() { 
      return numeric_limits<T>::quiet_NaN(); }
    static stan::agrad::fvar<T> signaling_NaN() { 
      return numeric_limits<T>::signaling_NaN(); }
    static stan::agrad::fvar<T> denorm_min() { 
      return numeric_limits<T>::denorm_min(); }

    static const bool is_iec559 = numeric_limits<T>::is_iec559;
    static const bool is_bounded = numeric_limits<T>::is_bounded;
    static const bool is_modulo = numeric_limits<T>::is_modulo;

    static const bool traps = numeric_limits<T>::traps;
    static const bool tinyness_before = numeric_limits<T>::tinyness_before;
    static const float_round_style round_style = 
      numeric_limits<T>::round_style;
  };
}
#endif
