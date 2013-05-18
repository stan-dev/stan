// Arguments: Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfNegBinomial : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 15;          // Failures/Counts
    param[1] = 50;          // Successes/Shape
    param[2] = 3;           // logit(p)/Inverse Scale
    parameters.push_back(param);
    cdf.push_back(0.4240861277); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // Successes/Shape
    index.push_back(1U);
    value.push_back(-1);
      
    // logit(p)/Inverse Scale
    index.push_back(2U);
    value.push_back(-1);
      
  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }

  template <typename T_n, typename T_shape, typename T_inv_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_shape, T_inv_scale>::type
  cdf(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::neg_binomial_cdf(n, alpha, beta);
  }


  template <typename T_n, typename T_shape, typename T_inv_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_shape, T_inv_scale>::type
  cdf_function(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    using std::log;
    using std::exp;
    using stan::math::binomial_coefficient_log;
    
    typename stan::return_type<T_shape, T_inv_scale>::type cdf(0);
      
    for (int i = 0; i <= n; i++) {
        
        typename stan::return_type<T_shape, T_inv_scale>::type temp;
        temp = binomial_coefficient_log(i + alpha - 1, i);

        cdf += exp( temp + alpha * log(beta / (1 + beta) ) + i * log(1 / (1 + beta) ) );
        
    }
      
    return cdf;
      
  }
};
