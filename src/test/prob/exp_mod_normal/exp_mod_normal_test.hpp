// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/exp_mod_normal_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionExpModNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-1.34102164500926350577078307323252902154767190882327); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-1.193147180559945397249); // expected log_prob

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 2.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-3.66695430596734551844348822271447899701975756228145); // expected log_prob

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

    //lambda
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
            typename T_inv_scale, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T_inv_scale& lambda, const T4&, const T5&) {
    return stan::math::exp_mod_normal_log(y, mu, sigma, lambda);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
            typename T_inv_scale, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_inv_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T_inv_scale& lambda, const T4&, const T5&) {
    return stan::math::exp_mod_normal_log<propto>(y, mu, sigma, lambda);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T_inv_scale, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_inv_scale>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                        const T_inv_scale& lambda, const T4&, const T5&) {
    return log(2) + log(lambda) 
      + lambda * (mu + 0.5 * lambda * sigma * sigma - y)
      + log(erfc((mu + lambda * sigma * sigma - y) / (sqrt(2.0) * sigma)));
  }
};

