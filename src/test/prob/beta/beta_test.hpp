// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/beta_log.hpp>

#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/fwd/core.hpp>
#include <boost/utility/enable_if.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsBeta : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.4;           // y
    param[1] = 0.5;           // alpha
    param[2] = 0.5;           // beta
    parameters.push_back(param);
    log_prob.push_back(-0.4311717080293276382896); // expected log_prob

    param[0] = 0.5;           // y
    param[1] = 2.0;          // alpha
    param[2] = 5.0;          // beta
    parameters.push_back(param);
    log_prob.push_back(-0.06453852113757105324332); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);
    
    index.push_back(0U);
    value.push_back(2.0);

    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_scale1, typename T_scale2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale1, T_scale2>::type 
  log_prob(const T_y& y, const T_scale1& alpha, const T_scale2& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::beta_log(y, alpha, beta);
  }

  template <bool propto, 
            typename T_y, typename T_scale1, typename T_scale2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale1, T_scale2>::type 
  log_prob(const T_y& y, const T_scale1& alpha, const T_scale2& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::beta_log<propto>(y, alpha, beta);
  }
  
  template <typename T_y, typename T_scale1, typename T_scale2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y,T_scale1,T_scale2,T3,T4,T5>::type
  log_prob_function(const T_y& y, const T_scale1& alpha,
                    const T_scale2& beta,
                    const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::log1m;
    
    return (alpha - 1.0) * log(y) + (beta - 1.0) * log1m(y) 
      + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta);
  }
};
