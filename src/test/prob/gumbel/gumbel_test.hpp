// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/gumbel_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionGumbel : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0); // expected log_prob

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-1.36787944117144232159552377016146086744581113103176783450783); // expected log_prob

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // beta
    parameters.push_back(param);
    log_prob.push_back(-5.38905609893065022723042746057500781318031557055184732408712); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // beta
    parameters.push_back(param);
    log_prob.push_back(-3.34108104263468429556956520337231251605487161479989511157723); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
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
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::gumbel_log(y, mu, beta);
  }

  template <bool propto, 
            typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::gumbel_log<propto>(y, mu, beta);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& beta,
                        const T3&, const T4&, const T5&) {

    return -log(beta) - (y - mu) / beta + exp((mu - y) / beta);
  }
};

