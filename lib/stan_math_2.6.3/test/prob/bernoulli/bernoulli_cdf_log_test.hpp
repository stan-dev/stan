// Arguments: Ints, Doubles
#include <stan/math/prim/scal/prob/bernoulli_cdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfLogBernoulli: public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf_log) {
    vector<double> param(2);

    param[0] = 0;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf_log.push_back(std::log(1 - param[1])); // expected cdf_log
      
    param[0] = 1;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf_log.push_back(0);       // expected cdf_log
      
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
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type
  cdf_log(const T_n& n, const T_prob& theta, const T2&,
          const T3&, const T4&, const T5&) {
    return stan::math::bernoulli_cdf_log(n, theta);
  }


  template <typename T_n, typename T_prob, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type
  cdf_log_function(const T_n& n, const T_prob& theta,const T2&,
                   const T3&, const T4&, const T5&) {

    if(n < 0) return stan::math::negative_infinity();
    if(n < 1) return log(1 - theta);
    else      return 0;
      
  }
};
