#ifndef __STAN__AGRAD__FVAR_HPP__
#define __STAN__AGRAD__FVAR_HPP__

//fvar
#include <stan/agrad/fwd/fvar.hpp>

//numeric limits
#include <stan/agrad/fwd/numeric_limits.hpp>

//comparison operators
#include <stan/agrad/fwd/operator_less_than.hpp>
#include <stan/agrad/fwd/operator_less_than_or_equal.hpp>
#include <stan/agrad/fwd/operator_greater_than.hpp>
#include <stan/agrad/fwd/operator_greater_than_or_equal.hpp>
#include <stan/agrad/fwd/operator_equal.hpp>
#include <stan/agrad/fwd/operator_not_equal.hpp>

//binary infix operators
#include <stan/agrad/fwd/operator_addition.hpp>
#include <stan/agrad/fwd/operator_subtraction.hpp>
#include <stan/agrad/fwd/operator_multiplication.hpp>
#include <stan/agrad/fwd/operator_division.hpp>

//unary prefix operators
#include <stan/agrad/fwd/operator_unary_minus.hpp>

//absolute functions
#include <stan/agrad/fwd/abs.hpp>
#include <stan/agrad/fwd/fabs.hpp>
#include <stan/agrad/fwd/fdim.hpp>

//bound functions
#include <stan/agrad/fwd/fmin.hpp>
#include <stan/agrad/fwd/fmax.hpp>

//arithmetic functions
#include <stan/agrad/fwd/fmod.hpp>

//rounding functions
#include <stan/agrad/fwd/floor.hpp>
#include <stan/agrad/fwd/ceil.hpp>
#include <stan/agrad/fwd/round.hpp>
#include <stan/agrad/fwd/trunc.hpp>

//power and log functions
#include <stan/agrad/fwd/sqrt.hpp>
#include <stan/agrad/fwd/cbrt.hpp>
#include <stan/agrad/fwd/square.hpp>
#include <stan/agrad/fwd/exp.hpp>
#include <stan/agrad/fwd/exp2.hpp>
#include <stan/agrad/fwd/log.hpp>
#include <stan/agrad/fwd/log2.hpp>
#include <stan/agrad/fwd/log10.hpp>
#include <stan/agrad/fwd/pow.hpp>

//trig functions
#include <stan/agrad/fwd/hypot.hpp>
#include <stan/agrad/fwd/cos.hpp>
#include <stan/agrad/fwd/sin.hpp>
#include <stan/agrad/fwd/cos.hpp>
#include <stan/agrad/fwd/tan.hpp>
#include <stan/agrad/fwd/acos.hpp>
#include <stan/agrad/fwd/asin.hpp>
#include <stan/agrad/fwd/atan.hpp>
#include <stan/agrad/fwd/atan2.hpp>

//hyperbolic trig functions
#include <stan/agrad/fwd/cosh.hpp>
#include <stan/agrad/fwd/sinh.hpp>
#include <stan/agrad/fwd/tanh.hpp>
#include <stan/agrad/fwd/asinh.hpp>
#include <stan/agrad/fwd/acosh.hpp>
#include <stan/agrad/fwd/atanh.hpp>

//link functions
#include <stan/agrad/fwd/logit.hpp>
#include <stan/agrad/fwd/inv_logit.hpp>
#include <stan/agrad/fwd/inv_cloglog.hpp>

//probability related functions
#include <stan/agrad/fwd/erf.hpp>
#include <stan/agrad/fwd/erfc.hpp>
#include <stan/agrad/fwd/phi.hpp>
#include <stan/agrad/fwd/binary_log_loss.hpp>

//combinatorial functions
#include <stan/agrad/fwd/tgamma.hpp>
#include <stan/agrad/fwd/lgamma.hpp>
#include <stan/agrad/fwd/lmgamma.hpp>
#include <stan/agrad/fwd/lbeta.hpp>
#include <stan/agrad/fwd/binomial_coefficient_log.hpp>

//composed functions
#include <stan/agrad/fwd/expm1.hpp>
#include <stan/agrad/fwd/fma.hpp>
#include <stan/agrad/fwd/log1p.hpp>
#include <stan/agrad/fwd/log1m.hpp>
#include <stan/agrad/fwd/log1p_exp.hpp>
#include <stan/agrad/fwd/log_sum_exp.hpp>
#include <stan/agrad/fwd/log_inv_logit.hpp>
#include <stan/agrad/fwd/log1m_inv_logit.hpp>

#endif
