#ifndef TEST__UNIT_DISTRIBUTION__UTILITY_HPP
#define TEST__UNIT_DISTRIBUTION__UTILITY_HPP

#include <vector>
#include <stan/math.hpp>
#include <stan/math/fwd/arr/fun/log_sum_exp.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/mat/fun/columns_dot_product.hpp>
#include <stan/math/fwd/mat/fun/columns_dot_self.hpp>
#include <stan/math/fwd/mat/fun/crossprod.hpp>
#include <stan/math/fwd/mat/fun/determinant.hpp>
#include <stan/math/fwd/mat/fun/divide.hpp>
#include <stan/math/fwd/mat/fun/dot_product.hpp>
#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/fwd/mat/fun/log_determinant.hpp>
#include <stan/math/fwd/mat/fun/log_softmax.hpp>
#include <stan/math/fwd/mat/fun/log_sum_exp.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/fwd/mat/fun/mdivide_left_tri_low.hpp>
#include <stan/math/fwd/mat/fun/mdivide_right.hpp>
#include <stan/math/fwd/mat/fun/mdivide_right_tri_low.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/fwd/mat/fun/qr_Q.hpp>
#include <stan/math/fwd/mat/fun/qr_R.hpp>
#include <stan/math/fwd/mat/fun/rows_dot_product.hpp>
#include <stan/math/fwd/mat/fun/rows_dot_self.hpp>
#include <stan/math/fwd/mat/fun/softmax.hpp>
#include <stan/math/fwd/mat/fun/sort_asc.hpp>
#include <stan/math/fwd/mat/fun/sort_desc.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/tcrossprod.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/fwd/mat/fun/trace_gen_quad_form.hpp>
#include <stan/math/fwd/mat/fun/trace_quad_form.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/scal/fun/Phi.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/acos.hpp>
#include <stan/math/fwd/scal/fun/acosh.hpp>
#include <stan/math/fwd/scal/fun/asin.hpp>
#include <stan/math/fwd/scal/fun/asinh.hpp>
#include <stan/math/fwd/scal/fun/atan.hpp>
#include <stan/math/fwd/scal/fun/atan2.hpp>
#include <stan/math/fwd/scal/fun/atanh.hpp>
#include <stan/math/fwd/scal/fun/bessel_first_kind.hpp>
#include <stan/math/fwd/scal/fun/bessel_second_kind.hpp>
#include <stan/math/fwd/scal/fun/binary_log_loss.hpp>
#include <stan/math/fwd/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/fwd/scal/fun/cbrt.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/cosh.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/erf.hpp>
#include <stan/math/fwd/scal/fun/erfc.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/exp2.hpp>
#include <stan/math/fwd/scal/fun/expm1.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/falling_factorial.hpp>
#include <stan/math/fwd/scal/fun/fdim.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/fma.hpp>
#include <stan/math/fwd/scal/fun/fmax.hpp>
#include <stan/math/fwd/scal/fun/fmin.hpp>
#include <stan/math/fwd/scal/fun/fmod.hpp>
#include <stan/math/fwd/scal/fun/gamma_p.hpp>
#include <stan/math/fwd/scal/fun/gamma_q.hpp>
#include <stan/math/fwd/scal/fun/grad_inc_beta.hpp>
#include <stan/math/fwd/scal/fun/hypot.hpp>
#include <stan/math/fwd/scal/fun/inc_beta.hpp>
#include <stan/math/fwd/scal/fun/inv.hpp>
#include <stan/math/fwd/scal/fun/inv_cloglog.hpp>
#include <stan/math/fwd/scal/fun/inv_logit.hpp>
#include <stan/math/fwd/scal/fun/inv_sqrt.hpp>
#include <stan/math/fwd/scal/fun/inv_square.hpp>
#include <stan/math/fwd/scal/fun/is_inf.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/fwd/scal/fun/lbeta.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/lmgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/log10.hpp>
#include <stan/math/fwd/scal/fun/log1m.hpp>
#include <stan/math/fwd/scal/fun/log1m_exp.hpp>
#include <stan/math/fwd/scal/fun/log1m_inv_logit.hpp>
#include <stan/math/fwd/scal/fun/log1p.hpp>
#include <stan/math/fwd/scal/fun/log1p_exp.hpp>
#include <stan/math/fwd/scal/fun/log2.hpp>
#include <stan/math/fwd/scal/fun/log_diff_exp.hpp>
#include <stan/math/fwd/scal/fun/log_falling_factorial.hpp>
#include <stan/math/fwd/scal/fun/log_inv_logit.hpp>
#include <stan/math/fwd/scal/fun/log_mix.hpp>
#include <stan/math/fwd/scal/fun/log_rising_factorial.hpp>
#include <stan/math/fwd/scal/fun/log_sum_exp.hpp>
#include <stan/math/fwd/scal/fun/logit.hpp>
#include <stan/math/fwd/scal/fun/modified_bessel_first_kind.hpp>
#include <stan/math/fwd/scal/fun/modified_bessel_second_kind.hpp>
#include <stan/math/fwd/scal/fun/multiply_log.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/owens_t.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/primitive_value.hpp>
#include <stan/math/fwd/scal/fun/rising_factorial.hpp>
#include <stan/math/fwd/scal/fun/round.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/sinh.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/square.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/tanh.hpp>
#include <stan/math/fwd/scal/fun/tgamma.hpp>
#include <stan/math/fwd/scal/fun/trunc.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>

using std::vector;
using stan::math::var;
using stan::math::fvar;
using stan::is_vector;
using stan::is_constant_struct;
using stan::scalar_type;

typedef stan::math::index_type<Eigen::Matrix<double,1,1> >::type size_type;

//------------------------------------------------------------

struct empty {};

template <typename T>
struct is_empty {
  enum { value = false };
};

template <>
struct is_empty<empty> {
  enum { value = true };
};

//------------------------------------------------------------

namespace std {
  std::ostream& operator<<(std::ostream& os, const vector<double>& param) {
    os << "(";
    for (size_t n = 0; n < param.size(); n++) {
      os << param[n];
      if (n < param.size()-1)
  os << ", ";
    }
    os << ")";
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const vector<var>& param) {
    os << "(";
    for (size_t n = 0; n < param.size(); n++) {
      os << param[n];
      if (n < param.size()-1)
  os << ", ";
    }
    os << ")";
    return os;
  }

}


//------------------------------------------------------------
// default template handles Eigen::Matrix
template <typename T>
T get_params(const vector<vector<double> >& parameters, const size_t p) {
  T param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size()) 
      param(n) = parameters[n][p];
  return param;
}

// handle empty
template <>
empty get_params<empty>(const vector<vector<double> >& /*parameters*/, const size_t /*p*/) {
  return empty();
}
// handle scalars
template<>
double get_params<double>(const vector<vector<double> >& parameters, const size_t p) {
  double param(0);
  if (p < parameters[0].size())
    param = parameters[0][p];
  return param;
}
template<>
var get_params<var>(const vector<vector<double> >& parameters, const size_t p) {
  var param(0);
  if (p < parameters[0].size())
    param = parameters[0][p];
  return param;
}
template<>
fvar<double> get_params<fvar<double> >(const vector<vector<double> >& parameters, const size_t p) {
  fvar<double> param(0);
  if (p < parameters[0].size()) {
    param = parameters[0][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<var> get_params<fvar<var> >(const vector<vector<double> >& parameters, const size_t p) {
  fvar<var> param(0);
  if (p < parameters[0].size()) {
    param = parameters[0][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<double> > get_params<fvar<fvar<double> > >(const vector<vector<double> >& parameters, const size_t p) {
  fvar<fvar<double> > param(0);
  if (p < parameters[0].size()) {
    param = parameters[0][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<var> > get_params<fvar<fvar<var> > >(const vector<vector<double> >& parameters, const size_t p) {
  fvar<fvar<var> > param(0);
  if (p < parameters[0].size()) {
    param = parameters[0][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
int get_params<int>(const vector<vector<double> >& parameters, const size_t p) {
  int param(0);
  if (p < parameters[0].size())
    param = (int)parameters[0][p];
  return param;
}
// handle vectors
template <>
vector<int> get_params<vector<int> >(const vector<vector<double> >& parameters, const size_t p) {
  vector<int> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size())
      param[n] = parameters[n][p];
  return param;
}
template <>
vector<double> get_params<vector<double> >(const vector<vector<double> >& parameters, const size_t p) {
  vector<double> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size())
      param[n] = parameters[n][p];
  return param;
}
template <>
vector<var> get_params<vector<var> >(const vector<vector<double> >& parameters, const size_t p) {
  vector<var> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size())
      param[n] = parameters[n][p];
  return param;
}
template <>
vector<fvar<double> > get_params<vector<fvar<double> > >(const vector<vector<double> >& parameters, const size_t p) {
  vector<fvar<double> > param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size()) {
      param[n] = parameters[n][p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<var> > get_params<vector<fvar<var> > >(const vector<vector<double> >& parameters, const size_t p) {
  vector<fvar<var> > param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size()) {
      param[n] = parameters[n][p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<double> > > get_params<vector<fvar<fvar<double> > > >(const vector<vector<double> >& parameters, const size_t p) {
  vector<fvar<fvar<double> > > param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size()) {
      param[n] = parameters[n][p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<var> > > get_params<vector<fvar<fvar<var> > > >(const vector<vector<double> >& parameters, const size_t p) {
  vector<fvar<fvar<var> > > param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    if (p < parameters[0].size()) {
      param[n] = parameters[n][p];
      param[n].d_ = 1.0;
    }
  return param;
}


//------------------------------------------------------------

// default template handles Eigen::Matrix
template <typename T>
T get_params(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  T param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size())
      param(i) = parameters[i][p];
  return param;
}

// handle empty
template <>
empty get_params<empty>(const vector<vector<double> >& /*parameters*/, const size_t /*n*/, const size_t /*p*/) {
  return empty();
}
// handle scalars
template<>
double get_params<double>(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  double param(0);
  if (p < parameters[0].size())
    param = parameters[n][p];
  return param;
}
template<>
var get_params<var>(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  var param(0);
  if (p < parameters[0].size())
    param = parameters[n][p];
  return param;
}
template<>
fvar<double> get_params<fvar<double> >(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  fvar<double> param(0);
  if (p < parameters[0].size()) {
    param = parameters[n][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<var> get_params<fvar<var> >(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  fvar<var> param(0);
  if (p < parameters[0].size()) {
    param = parameters[n][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<double> > get_params<fvar<fvar<double> > >(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  fvar<fvar<double> > param(0);
  if (p < parameters[0].size()) {
    param = parameters[n][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<var> > get_params<fvar<fvar<var> > >(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  fvar<fvar<var> > param(0);
  if (p < parameters[0].size()) {
    param = parameters[n][p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
int get_params<int>(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  int param(0);
  if (p < parameters[0].size())
    param = (int)parameters[n][p];
  return param;
}
// handle vectors
template <>
vector<int> get_params<vector<int> >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<int> param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size())
      param[i] = parameters[i][p];
  return param;
}
template <>
vector<double> get_params<vector<double> >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<double> param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size())
      param[i] = parameters[i][p];
  return param;
}
template <>
vector<var> get_params<vector<var> >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<var> param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size())
      param[i] = parameters[i][p];
  return param;
}
template <>
vector<fvar<double> > get_params<vector<fvar<double> > >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<fvar<double> > param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size()) {
      param[i] = parameters[i][p];
      param[i].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<var> > get_params<vector<fvar<var> > >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<fvar<var> > param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size()) {
      param[i] = parameters[i][p];
      param[i].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<double> > > get_params<vector<fvar<fvar<double> > > >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<fvar<fvar<double> > > param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size()) {
      param[i] = parameters[i][p];
      param[i].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<var> > > get_params<vector<fvar<fvar<var> > > >(const vector<vector<double> >& parameters, const size_t /*n*/, const size_t p) {
  vector<fvar<fvar<var> > > param(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++)
    if (p < parameters[0].size()) {
      param[i] = parameters[i][p];
      param[i].d_ = 1.0;
    }
  return param;
}


//------------------------------------------------------------
// default template handles Eigen::Matrix
template <typename T>
T get_repeated_params(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  T param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++) {
    if (p < parameters.size())
      param(n) = parameters[p];
    else
      param(0) = 0;
  }
  return param;
}

// handle empty
template <>
empty get_repeated_params<empty>(const vector<double>& /*parameters*/, const size_t /*p*/, const size_t /*N_REPEAT*/) {
  return empty();
}
// handle scalars
template<>
double get_repeated_params<double>(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  double param(0);
  if (p < parameters.size())
    param = parameters[p];
  return param;
}
template<>
var get_repeated_params<var>(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  var param(0);
  if (p < parameters.size())
    param = parameters[p];
  return param;
}
template<>
fvar<double> get_repeated_params<fvar<double> >(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  fvar<double> param(0);
  if (p < parameters.size()) {
    param = parameters[p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<var> get_repeated_params<fvar<var> >(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  fvar<var> param(0);
  if (p < parameters.size()) {
    param = parameters[p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<double> > get_repeated_params<fvar<fvar<double> > >(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  fvar<fvar<double> > param(0);
  if (p < parameters.size()) {
    param = parameters[p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
fvar<fvar<var> > get_repeated_params<fvar<fvar<var> > >(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  fvar<fvar<var> > param(0);
  if (p < parameters.size()) {
    param = parameters[p];
    param.d_ = 1.0;
  }
  return param;
}
template<>
int get_repeated_params<int>(const vector<double>& parameters, const size_t p, const size_t /*N_REPEAT*/) {
  int param(0);
  if (p < parameters.size())
    param = (int)parameters[p];
  return param;
}
// handle vectors
template <>
vector<int> get_repeated_params<vector<int> >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<int> param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size())
      param[n] = parameters[p];
  return param;
}
template <>
vector<double> get_repeated_params<vector<double> >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<double> param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size())
      param[n] = parameters[p];
  return param;
}
template <>
vector<var> get_repeated_params<vector<var> >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<var> param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size())
      param[n] = parameters[p];
  return param;
}

template <>
vector<fvar<double> > get_repeated_params<vector<fvar<double> > >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<fvar<double> > param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size()) {
      param[n] = parameters[p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<var> > get_repeated_params<vector<fvar<var> > >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<fvar<var> > param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size()) {
      param[n] = parameters[p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<double> > > get_repeated_params<vector<fvar<fvar<double> > > >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<fvar<fvar<double> > > param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size()) {
      param[n] = parameters[p];
      param[n].d_ = 1.0;
    }
  return param;
}
template <>
vector<fvar<fvar<var> > > get_repeated_params<vector<fvar<fvar<var> > > >(const vector<double>& parameters, const size_t p, const size_t N_REPEAT) {
  vector<fvar<fvar<var> > > param(N_REPEAT);
  for (size_t n = 0; n < N_REPEAT; n++)
    if (p < parameters.size()) {
      param[n] = parameters[p];
      param[n].d_ = 1.0;
    }
  return param;
}

//------------------------------------------------------------

template <typename T>
T get_param(const vector<double>& params, const size_t n) {
  T param = 0;
  if (n < params.size())
    param = params[n];
  return param;
}

template <>
empty get_param<empty>(const vector<double>& /*params*/, const size_t /*n*/) {
  return empty();
}
template <>
fvar<double> get_param<fvar<double> >(const vector<double>& params, 
                                      const size_t n) {
  fvar<double> param = 0;
  if (n < params.size()) {
    param = params[n];
    param.d_ = 1.0;
  }
  return param;
}
template <>
fvar<var> get_param<fvar<var> >(const vector<double>& params, 
                                const size_t n) {
  fvar<var> param = 0;
  if (n < params.size()) {
    param = params[n];
    param.d_ = 1.0;
  }
  return param;
}
template <>
fvar<fvar<double> > get_param<fvar<fvar<double> > >(const vector<double>& params, 
                                                    const size_t n) {
  fvar<fvar<double> > param = 0;
  if (n < params.size()) {
    param = params[n];
    param.d_ = 1.0;
  }
  return param;
}
template <>
fvar<fvar<var> > get_param<fvar<fvar<var> > >(const vector<double>& params, 
                                              const size_t n) {
  fvar<fvar<var> > param = 0;
  if (n < params.size()) {
    param = params[n];
    param.d_ = 1.0;
  }
  return param;
}

//------------------------------------------------------------

template <typename T>
typename scalar_type<T>::type
select_var_param(const vector<vector<double> >& parameters, const size_t n, const size_t p) {
  typename scalar_type<T>::type param(0);
  if (p < parameters[0].size()) {
    if (is_vector<T>::value && !is_constant_struct<T>::value)
      param = parameters[n][p];
    else
      param = parameters[0][p];
  }
  return param;
}

template <>
empty select_var_param<empty>(const vector<vector<double> >& /*parameters*/, const size_t /*n*/, const size_t /*p*/) {
  return empty();
}


//------------------------------------------------------------
template <typename T0, typename T1, typename T2,
    typename T3, typename T4, typename T5>
struct all_scalar {
  enum {
    value = (!is_vector<T0>::value || is_empty<T0>::value)
    && (!is_vector<T1>::value || is_empty<T1>::value)
    && (!is_vector<T2>::value || is_empty<T2>::value)
    && (!is_vector<T3>::value || is_empty<T3>::value)
    && (!is_vector<T4>::value || is_empty<T4>::value)
    && (!is_vector<T5>::value || is_empty<T5>::value)
  };
};

template <typename T0, typename T1, typename T2,
    typename T3, typename T4, typename T5>
struct all_constant {
  enum {
    value = (is_constant_struct<T0>::value || is_empty<T0>::value)
    && (is_constant_struct<T1>::value || is_empty<T1>::value)
    && (is_constant_struct<T2>::value || is_empty<T2>::value)
    && (is_constant_struct<T3>::value || is_empty<T3>::value)
    && (is_constant_struct<T4>::value || is_empty<T4>::value)
    && (is_constant_struct<T5>::value || is_empty<T5>::value)
  };
};

template <typename T0, typename T1, typename T2,
    typename T3, typename T4, typename T5>
struct all_var {
  enum {
    value = (!is_constant_struct<T0>::value || is_empty<T0>::value)
    && (!is_constant_struct<T1>::value || is_empty<T1>::value)
    && (!is_constant_struct<T2>::value || is_empty<T2>::value)
    && (!is_constant_struct<T3>::value || is_empty<T3>::value)
    && (!is_constant_struct<T4>::value || is_empty<T4>::value)
    && (!is_constant_struct<T5>::value || is_empty<T5>::value)
  };
};


template <typename T0, typename T1, typename T2,
    typename T3, typename T4, typename T5>
struct any_vector {
  enum {
    value = is_vector<T0>::value
    || is_vector<T1>::value
    || is_vector<T2>::value
    || is_vector<T3>::value
    || is_vector<T4>::value
    || is_vector<T5>::value
  };
};

//------------------------------------------------------------
template <typename T>
void add_var(vector<var>& /*x*/, T& /*p*/) {
}

template <>
void add_var<var>(vector<var>& x, var& p) {
  x.push_back(p);
}

template <>
void add_var<vector<var> >(vector<var>& x, vector<var>& p) {
  x.insert(x.end(), p.begin(), p.end());
}

template <>
void add_var<Eigen::Matrix<var, 1, Eigen::Dynamic> >(vector<var>& x, Eigen::Matrix<var, 1, Eigen::Dynamic>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n));
}

template <>
void add_var<Eigen::Matrix<var, Eigen::Dynamic, 1> >(vector<var>& x, Eigen::Matrix<var, Eigen::Dynamic, 1>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n));
}

template <>
void add_var<fvar<var> >(vector<var>& x, fvar<var>& p) {
  x.push_back(p.val_);
}

template <>
void add_var<vector<fvar<var> > >(vector<var>& x, vector<fvar<var> >& p) {
  for (size_t n = 0; n < p.size(); n++)
    x.push_back(p[n].val_);}

template <>
void add_var<Eigen::Matrix<fvar<var>, 1, Eigen::Dynamic> >(vector<var>& x, Eigen::Matrix<fvar<var>, 1, Eigen::Dynamic>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n).val_);
}

template <>
void add_var<Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> >(vector<var>& x, Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n).val_);
}

template <>
void add_var<fvar<fvar<var> > >(vector<var>& x, fvar<fvar<var> >& p) {
  x.push_back(p.val_.val_);
}

template <>
void add_var<vector<fvar<fvar<var> > > >(vector<var>& x, vector<fvar<fvar<var> > >& p) {
  for (size_t n = 0; n < p.size(); n++)
    x.push_back(p[n].val_.val_);
}

template <>
void add_var<Eigen::Matrix<fvar<fvar<var> >, 1, Eigen::Dynamic> >(vector<var>& x, Eigen::Matrix<fvar<fvar<var> >, 1, Eigen::Dynamic>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n).val_.val_);
}

template <>
void add_var<Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1> >(vector<var>& x, Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1>& p) {
  for (size_type n = 0; n < p.size(); n++)
    x.push_back(p(n).val_.val_);
}

template <typename T0, typename T1, typename T2,
          typename T3, typename T4, typename T5>
void add_vars(vector<var>& x, T0& p0, T1& p1, T2& p2, 
              T3& p3, T4& p4, T5& p5) {
  if (!is_constant_struct<T0>::value)
    add_var(x, p0);
  if (!is_constant_struct<T1>::value)
    add_var(x, p1);
  if (!is_constant_struct<T2>::value)
    add_var(x, p2);
  if (!is_constant_struct<T3>::value)
    add_var(x, p3);
  if (!is_constant_struct<T4>::value)
    add_var(x, p4);
  if (!is_constant_struct<T5>::value)
    add_var(x, p5);
}
//------------------------------------------------------------




#endif

