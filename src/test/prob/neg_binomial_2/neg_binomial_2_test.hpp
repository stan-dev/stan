// Arguments: Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/neg_binomial_2_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsNegBinomial2 : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // mu
    param[2] = 1.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-5.55873452880806); // expected log_prob

    param[0] = 100;          // n
    param[1] = 3.0;          // mu
    param[2] = 3.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-69.1303554183435); // expected log_prob

    param[0] = 100;          // n
    param[1] = 4;          // mu
    param[2] = 200;          // phi
    parameters.push_back(param);
    log_prob.push_back(-209.614066642678); // expected log_prob
  }

  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // mu
    index.push_back(1U);
    value.push_back(-1);

    // phi
    index.push_back(2U);
    value.push_back(-1);
  }

  template <class T_n, class T_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_location& mu, const T_inv_scale& phi,
           const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_2_log(n, mu, phi);
  }

  template <bool propto,
            class T_n, class T_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_location& mu, const T_inv_scale& phi,
           const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_2_log<propto>(n, mu, phi);
  }


  template <class T_n, class T_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_location,T_inv_scale>::type
  log_prob_function(const T_n& n, const T_location& mu, 
                    const T_inv_scale& phi, const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::multiply_log;

    if (n != 0)
      return binomial_coefficient_log<typename stan::scalar_type<T_inv_scale>::type> (n + phi - 1.0, n)
        + multiply_log(n, mu) + multiply_log(phi, phi) - (n+phi)*log(mu + phi);
    return log(0.0);
  }
};


