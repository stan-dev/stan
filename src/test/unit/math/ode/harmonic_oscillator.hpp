#ifndef HARMONIC_OSCILLATOR
#define HARMONIC_OSCILLATOR

#include <stan/meta/traits.hpp>

template <typename T0, typename T1, typename T2>
inline
std::vector<typename stan::return_type<T1,T2>::type> 
harm_osc_ode(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int) { // integer data
  std::vector<typename stan::return_type<T1,T2>::type> res;
  res.push_back(y_in[1]);
  res.push_back(-y_in[0] - theta[0]*y_in[1]);

  return res;
}

struct harm_osc_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline 
  std::vector<typename stan::return_type<T1,T2>::type> 
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int,
             std::ostream* msgs) const { // integer data
    return harm_osc_ode(t_in, y_in, theta, x, x_int);
  }
};

#endif
