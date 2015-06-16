// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/pareto_type_2_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionParetoType2 : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 1.5;           // y
    param[1] = 0.1;           // mu
    param[2] = 0.5;           // lambda
    param[3] = 3.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-3.548244797701304964476776149240); // expected log_prob

    param[0] = 19.5;          // y
    param[1] = 0;          // mu
    param[2] = 0.15;          // lambda
    param[3] = 5.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-25.74462604188692758825580639457); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // lambda
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    // alpha
    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());
  }


  template <class T_y, class T_loc, class T_scale, class T_shape, 
            typename T4, typename T5>
  typename stan::return_type<T_y, T_scale, T_shape,T_loc>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& lambda, 
           const T_shape& alpha, const T4&, const T5&) {
    return stan::math::pareto_type_2_log(y, mu, lambda, alpha);
  }

  template <bool propto, 
            class T_y, class T_loc, class T_scale, class T_shape,
            typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& lambda,
           const T_shape& alpha, const T4&, const T5&) {
    return stan::math::pareto_type_2_log<propto>(y, mu, lambda, alpha);
  }
  

  template <class T_y, class T_loc, class T_scale, class T_shape,
            typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& lambda, 
                    const T_shape& alpha, const T4&, const T5&) {
    
    return log(alpha) - log(lambda) - (alpha + 1.0) * log1p((y - mu) / lambda);
  }
};

