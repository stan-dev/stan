// Arguments: Int, Double
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>
#include <stan/agrad/special_functions.hpp>

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

    param[0] = 192;          // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);
    cdf.push_back(1.0);      // expected cdf

    param[0] = 0.0;          // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);
    cdf.push_back(0.04978707); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }
  
  bool has_lower_bound() {
    return true;
  }
  
  double lower_bound() {
    return 0.0;
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
	    typename T9,
	    typename Policy>
  typename stan::return_type<T_rate>::type
  cdf(const T_n& n, const T_rate& lambda, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::poisson_cdf(n, lambda, Policy());
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
    using stan::agrad::tgamma;
    using boost::math::tgamma;
    using std::exp;
    using stan::agrad::exp;
    
    typename stan::return_type<T_rate>::type cdf(0);
    for (int i = 0; i <= n; i++) {
      cdf += pow(lambda, i) / tgamma(i+1);
    }
    cdf *= exp(-lambda);
    return cdf;
  }
};
