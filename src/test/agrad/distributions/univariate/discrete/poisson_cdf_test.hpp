// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfPoisson : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = 13.0;         // lambda
    parameters.push_back(param);
    cdf.push_back(0.890465); // expected cdf

    param[0] = 82;           // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);
    cdf.push_back(0.99999998); // expected cdf
    
    param[0] = 0.0;          // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);
    cdf.push_back(0.04978707); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }

  template <typename T_n, typename T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_rate>::type
  cdf(const T_n& n, const T_rate& lambda, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::poisson_cdf(n, lambda);
  }


  template <typename T_n, typename T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_rate>::type
  cdf_function(const T_n& n, const T_rate& lambda, const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::pow;
    using stan::agrad::pow;
    using stan::agrad::lgamma;
    using boost::math::lgamma;
    using std::exp;
    using stan::agrad::exp;
    using std::log;
    
    typename stan::return_type<T_rate>::type cdf(0);
    for (int i = 0; i <= n; i++) {
      cdf += exp(i * log(lambda) - lgamma(i+1));
    }
    cdf *= exp(-lambda);
    return cdf;
  }
};
