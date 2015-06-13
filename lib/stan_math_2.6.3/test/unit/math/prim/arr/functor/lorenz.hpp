#ifndef TEST__UNIT__MATH__ODE__LORENZ_HPP
#define TEST__UNIT__MATH__ODE__LORENZ_HPP

template <typename T0, typename T1, typename T2>
inline
std::vector<typename stan::return_type<T1,T2>::type> 
lorenz_ode(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int) { // integer data
  std::vector<typename stan::return_type<T1,T2>::type> res;
  res.push_back(theta.at(0)*(y_in.at(1) - y_in.at(0)));
  res.push_back(theta.at(1)*y_in.at(0) - y_in.at(1) - y_in.at(0)*y_in.at(2));
  res.push_back(-theta.at(2)*y_in.at(2) + y_in.at(0)*y_in.at(1));
  return res;
}

struct lorenz_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline 
  std::vector<typename stan::return_type<T1,T2>::type> 
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int,
             std::ostream* msgs) const { // integer data
    return lorenz_ode(t_in, y_in, theta, x, x_int);
  }
};

#endif
