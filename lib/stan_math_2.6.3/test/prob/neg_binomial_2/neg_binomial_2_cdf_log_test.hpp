// Arguments: Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/neg_binomial_2_cdf_log.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfLogNegBinomial2 : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf_log) {
    vector<double> param(3);

    param[0] = 3;          // n
    param[1] = 10;          // mu
    param[2] = 20;           // phi
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.0264752601628231235)); // expected cdf_log
    
    param[0] = 7;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.091899254171238523026)); // expected cdf_log
    
    param[0] = 0;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.0001048576000000001529)); // expected cdf_log
    
    param[0] = 1;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.00073400320000000126002)); // expected cdf_log
    
    param[0] = 0;          // n
    param[1] = 10;          // mu
    param[2] = 1;           // phi
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.090909090909090897736)); // expected cdf_log
    
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // mu
    index.push_back(1U);
    value.push_back(-1);
      
    // phi
    index.push_back(2U);
    value.push_back(-1);
      
  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }
  
  template <typename T_n, typename T_location, typename T_precision,
      typename T3, typename T4, typename T5>
  typename stan::return_type<T_location, T_precision>::type
  cdf_log(const T_n& n, const T_location& alpha, const T_precision& beta,
          const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_2_cdf_log(n, alpha, beta);
  }


  template <typename T_n, typename T_location, typename T_precision,
      typename T3, typename T4, typename T5>
  typename stan::return_type<T_location, T_precision>::type
  cdf_log_function(const T_n& nn, const T_location& mu, const T_precision& phi,
                   const T3&, const T4&, const T5&) {

    using std::log;
    using std::exp;
    using stan::math::binomial_coefficient_log;
    using stan::math::multiply_log;
    
    typename stan::return_type<T_location, T_precision>::type cdf(0);
    
    for (int n = 0; n <= nn; n++) {
      typename stan::return_type<T_location, T_precision>::type lp(0);
      if (n != 0)
        lp += binomial_coefficient_log<typename stan::scalar_type<T_precision>::type>
            (n + phi - 1.0, n);
      lp +=  multiply_log(n, mu) + multiply_log(phi, phi) - (n+phi)*log(mu + phi);
      cdf += exp(lp);
    }
      
    return log(cdf);
      
  }
};
