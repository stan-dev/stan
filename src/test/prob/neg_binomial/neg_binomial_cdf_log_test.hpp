// Arguments: Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/neg_binomial_cdf_log.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfLogNegBinomial : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(3);

    param[0] = 15;          // Failures/Counts
    param[1] = 50;          // Successes/Shape
    param[2] = 3;           // logit(p)/Inverse Scale
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.4240861277740262114122)); // expected cdf_log
    
    param[0] = 0;          // Failures/Counts
    param[1] = 15;          // Successes/Shape
    param[2] = 3;           // logit(p)/Inverse Scale
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.013363461010158063716)); // expected cdf_log

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
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_shape, T_inv_scale>::type
  cdf_log(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
          const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_cdf_log(n, alpha, beta);
  }


  template <typename T_n, typename T_shape, typename T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_shape, T_inv_scale>::type
  cdf_log_function(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
                   const T3&, const T4&, const T5&) {

    using std::log;
    using std::exp;
    using stan::math::binomial_coefficient_log;
    
    typename stan::return_type<T_shape, T_inv_scale>::type cdf(0);
      
    for (int i = 0; i <= n; i++) {
        typename stan::return_type<T_shape, T_inv_scale>::type temp;
        temp = binomial_coefficient_log(i + alpha - 1, i);

        cdf += exp( temp + alpha * log(beta / (1 + beta) ) + i * log(1 / (1 + beta) ) );
    }
      
    return log(cdf);
  }
};
