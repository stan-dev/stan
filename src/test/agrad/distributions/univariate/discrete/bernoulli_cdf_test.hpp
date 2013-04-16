// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfBernoulli: public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 0;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf.push_back(1 - param[1]); // expected cdf
      
    param[0] = 1;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf.push_back(1);       // expected cdf
      
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
      
    // p (Probability)
    index.push_back(1U);
    value.push_back(-1e-4);

    index.push_back(1U);
    value.push_back(1+1e-4);
  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }

  template <typename T_n, typename T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  cdf(const T_n& n, const T_prob& theta, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_cdf(n, theta);
  }


  template <typename T_n, typename T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  cdf_function(const T_n& n, const T_prob& theta,const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    if(n < 0) return 0;
    if(n < 1) return 1 - theta;
    else      return 1;
      
  }
};
