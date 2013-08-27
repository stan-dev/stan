#ifndef __STAN__DIFF__FVAR_HPP__
#define __STAN__DIFF__FVAR_HPP__

//fvar
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/traits.hpp>

//numeric limits
#include <stan/diff/fwd/numeric_limits.hpp>

//comparison operators
#include <stan/diff/fwd/operator_less_than.hpp>
#include <stan/diff/fwd/operator_less_than_or_equal.hpp>
#include <stan/diff/fwd/operator_greater_than.hpp>
#include <stan/diff/fwd/operator_greater_than_or_equal.hpp>
#include <stan/diff/fwd/operator_equal.hpp>
#include <stan/diff/fwd/operator_not_equal.hpp>

//binary infix operators
#include <stan/diff/fwd/operator_addition.hpp>
#include <stan/diff/fwd/operator_subtraction.hpp>
#include <stan/diff/fwd/operator_multiplication.hpp>
#include <stan/diff/fwd/operator_division.hpp>

//unary prefix operators
#include <stan/diff/fwd/operator_unary_minus.hpp>

//absolute functions
#include <stan/diff/fwd/abs.hpp>
#include <stan/diff/fwd/fabs.hpp>
#include <stan/diff/fwd/fdim.hpp>

//bound functions
#include <stan/diff/fwd/fmin.hpp>
#include <stan/diff/fwd/fmax.hpp>

//arithmetic functions
#include <stan/diff/fwd/fmod.hpp>

//rounding functions
#include <stan/diff/fwd/floor.hpp>
#include <stan/diff/fwd/ceil.hpp>
#include <stan/diff/fwd/round.hpp>
#include <stan/diff/fwd/trunc.hpp>

//power and log functions
#include <stan/diff/fwd/sqrt.hpp>
#include <stan/diff/fwd/cbrt.hpp>
#include <stan/diff/fwd/square.hpp>
#include <stan/diff/fwd/exp.hpp>
#include <stan/diff/fwd/exp2.hpp>
#include <stan/diff/fwd/log.hpp>
#include <stan/diff/fwd/log2.hpp>
#include <stan/diff/fwd/log10.hpp>
#include <stan/diff/fwd/pow.hpp>
#include <stan/diff/fwd/inv.hpp>
#include <stan/diff/fwd/inv_sqrt.hpp>
#include <stan/diff/fwd/inv_square.hpp>

//trig functions
#include <stan/diff/fwd/hypot.hpp>
#include <stan/diff/fwd/cos.hpp>
#include <stan/diff/fwd/sin.hpp>
#include <stan/diff/fwd/cos.hpp>
#include <stan/diff/fwd/tan.hpp>
#include <stan/diff/fwd/acos.hpp>
#include <stan/diff/fwd/asin.hpp>
#include <stan/diff/fwd/atan.hpp>
#include <stan/diff/fwd/atan2.hpp>

//hyperbolic trig functions
#include <stan/diff/fwd/cosh.hpp>
#include <stan/diff/fwd/sinh.hpp>
#include <stan/diff/fwd/tanh.hpp>
#include <stan/diff/fwd/asinh.hpp>
#include <stan/diff/fwd/acosh.hpp>
#include <stan/diff/fwd/atanh.hpp>

//link functions
#include <stan/diff/fwd/logit.hpp>
#include <stan/diff/fwd/inv_logit.hpp>
#include <stan/diff/fwd/inv_cloglog.hpp>

//probability related functions
#include <stan/diff/fwd/erf.hpp>
#include <stan/diff/fwd/erfc.hpp>
#include <stan/diff/fwd/phi.hpp>
#include <stan/diff/fwd/binary_log_loss.hpp>

//combinatorial functions
#include <stan/diff/fwd/tgamma.hpp>
#include <stan/diff/fwd/lgamma.hpp>
#include <stan/diff/fwd/lmgamma.hpp>
#include <stan/diff/fwd/gamma_p.hpp>
#include <stan/diff/fwd/gamma_q.hpp>
#include <stan/diff/fwd/lbeta.hpp>
#include <stan/diff/fwd/binomial_coefficient_log.hpp>
#include <stan/diff/fwd/bessel_first_kind.hpp>
#include <stan/diff/fwd/bessel_second_kind.hpp>
#include <stan/diff/fwd/modified_bessel_first_kind.hpp>
#include <stan/diff/fwd/modified_bessel_second_kind.hpp>
#include <stan/diff/fwd/falling_factorial.hpp>
#include <stan/diff/fwd/rising_factorial.hpp>
#include <stan/diff/fwd/log_rising_factorial.hpp>
#include <stan/diff/fwd/log_falling_factorial.hpp>

//composed functions
#include <stan/diff/fwd/expm1.hpp>
#include <stan/diff/fwd/fma.hpp>
#include <stan/diff/fwd/log1m.hpp>
#include <stan/diff/fwd/log1p.hpp>
#include <stan/diff/fwd/log1m_exp.hpp>
#include <stan/diff/fwd/log1p_exp.hpp>
#include <stan/diff/fwd/log_diff_exp.hpp>
#include <stan/diff/fwd/log_sum_exp.hpp>
#include <stan/diff/fwd/log_inv_logit.hpp>
#include <stan/diff/fwd/log1m_inv_logit.hpp>

#endif
