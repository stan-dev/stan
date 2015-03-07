struct mock_ode_functor {
  template <typename T0, typename T1, typename T2>
  inline 
  std::vector<typename stan::return_type<T1,T2>::type> 
  operator()(const T0& t_in, 
             const std::vector<T1>& y_in,
             const std::vector<T2>& theta,
             const std::vector<double>& x, 
             const std::vector<int>& x_int,
             std::ostream* msgs) const { 
    std::vector<typename stan::return_type<T1,T2>::type> states;
    states = y_in;
    return states;
  }
};
