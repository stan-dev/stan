// Arguments: Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/neg_binomial_2.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLogNegBinomial2 : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& ccdf) {
    vector<double> param(3);

    param[0] = 3;          // n
    param[1] = 10;          // mu
    param[2] = 20;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0 - 0.02647526)); // expected ccdf
    
    param[0] = 7;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0 - 0.09189925)); // expected ccdf
    
    param[0] = 0;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0 - 0.0001048576)); // expected ccdf
    
    param[0] = 1;          // n
    param[1] = 15;          // mu
    param[2] = 10;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0 - 0.0007340032)); // expected ccdf
    
    param[0] = 0;          // n
    param[1] = 10;          // mu
    param[2] = 1;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0 - 0.09090909)); // expected ccdf
    
    param[0] = -1;          // n
    param[1] = 10;          // mu
    param[2] = 1;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0)); // expected ccdf
    
    param[0] = -6.0;          // n
    param[1] = 10;          // mu
    param[2] = 1;           // phi
    parameters.push_back(param);
    ccdf.push_back(log(1.0)); // expected ccdf
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
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_location, T_precision>::type
  ccdf_log(const T_n& n, const T_location& alpha, const T_precision& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::neg_binomial_2_ccdf_log(n, alpha, beta);
  }


  template <typename T_n, typename T_location, typename T_precision,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var
  ccdf_log_function(const T_n& nn, const T_location& mu, const T_precision& phi,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    using std::log;
    using std::exp;
    using stan::math::binomial_coefficient_log;
    using stan::math::multiply_log;
    
    var ccdf(0);
    
    for (int n = 0; n <= nn; n++) {
      var lp(0);
      if (n != 0)
        lp += binomial_coefficient_log<typename stan::scalar_type<T_precision>::type>
            (n + phi - 1.0, n);
      lp +=  multiply_log(n, mu) + multiply_log(phi, phi) - (n+phi)*log(mu + phi);
      ccdf += exp(lp);
    }
      
    return log(1.0 - ccdf);
      
  }
};
