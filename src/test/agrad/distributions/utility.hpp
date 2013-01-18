#ifndef __TEST__AGRAD__DISTRIBUTIONS__UTILITY_HPP__
#define __TEST__AGRAD__DISTRIBUTIONS__UTILITY_HPP__

#include <boost/math/policies/policy.hpp>
#include <vector>

using std::vector;
using stan::agrad::var;
using stan::is_vector;
using stan::is_constant;

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

typedef stan::math::default_policy default_policy;

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

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




//------------------------------------------------------------
template <typename T0, typename T1, typename T2,
	  typename T3, typename T4, typename T5, 
	  typename T6, typename T7, typename T8, 
	  typename T9>
struct all_scalar {
  enum {
    value = (!is_vector<T0>::value || is_empty<T0>::value)
    && (!is_vector<T1>::value || is_empty<T1>::value)
    && (!is_vector<T2>::value || is_empty<T2>::value)
    && (!is_vector<T3>::value || is_empty<T3>::value)
    && (!is_vector<T4>::value || is_empty<T4>::value)
    && (!is_vector<T5>::value || is_empty<T5>::value)
    && (!is_vector<T6>::value || is_empty<T6>::value)
    && (!is_vector<T7>::value || is_empty<T7>::value)
    && (!is_vector<T8>::value || is_empty<T8>::value)
    && (!is_vector<T9>::value || is_empty<T9>::value)
  };
};

template <typename T0, typename T1, typename T2,
	  typename T3, typename T4, typename T5, 
	  typename T6, typename T7, typename T8, 
	  typename T9>
struct all_constant {
  enum {
    value = (is_constant<T0>::value || is_empty<T0>::value)
    && (is_constant<T1>::value || is_empty<T1>::value)
    && (is_constant<T2>::value || is_empty<T2>::value)
    && (is_constant<T3>::value || is_empty<T3>::value)
    && (is_constant<T4>::value || is_empty<T4>::value)
    && (is_constant<T5>::value || is_empty<T5>::value)
    && (is_constant<T6>::value || is_empty<T6>::value)
    && (is_constant<T7>::value || is_empty<T7>::value)
    && (is_constant<T8>::value || is_empty<T8>::value)
    && (is_constant<T9>::value || is_empty<T9>::value)
  };
};

template <typename T0, typename T1, typename T2,
	  typename T3, typename T4, typename T5, 
	  typename T6, typename T7, typename T8, 
	  typename T9>
struct all_var {
  enum {
    value = (!is_constant<T0>::value || is_empty<T0>::value)
    && (!is_constant<T1>::value || is_empty<T1>::value)
    && (!is_constant<T2>::value || is_empty<T2>::value)
    && (!is_constant<T3>::value || is_empty<T3>::value)
    && (!is_constant<T4>::value || is_empty<T4>::value)
    && (!is_constant<T5>::value || is_empty<T5>::value)
    && (!is_constant<T6>::value || is_empty<T6>::value)
    && (!is_constant<T7>::value || is_empty<T7>::value)
    && (!is_constant<T8>::value || is_empty<T8>::value)
    && (!is_constant<T9>::value || is_empty<T9>::value)
  };
};


#endif

