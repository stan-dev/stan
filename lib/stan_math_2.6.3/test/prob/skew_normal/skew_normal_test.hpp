// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/skew_normal_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionSkewNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.9189385332046727805633); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.8985451316681772881978); // expected log_prob

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 2.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-12.58589283917201839813); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    param[3] = 2.9;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-6.693233548678988675817); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    //alpha
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
            typename T_shape, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T_shape& alpha, const T4&, const T5&) {
    return stan::math::skew_normal_log(y, mu, sigma, alpha);
  }

  template <bool propto, 
            typename T_y, typename T_loc, typename T_scale,
            typename T_shape, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T_shape& alpha, const T4&, const T5&) {
    return stan::math::skew_normal_log<propto>(y, mu, sigma, alpha);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T_shape, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T_shape& alpha, const T4&, const T5&) {
    return -0.5 * log(2.0 * stan::math::pi()) 
      - log(sigma) - (y - mu) / sigma * (y - mu) / sigma * 0.5 
      + log(erfc(-alpha * (y - mu) / (sigma * std::sqrt(2.0))));
  }
};

