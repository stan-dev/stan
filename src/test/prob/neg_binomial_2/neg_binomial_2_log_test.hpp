// Arguments: Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/neg_binomial_2_log_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsNegBinomial2Log : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // eta
    param[2] = 1.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-3.20887218205076511331179862116263121414716429208289234888793); // expected log_prob

    param[0] = 100;          // n
    param[1] = -3.0;          // eta
    param[2] = 3.5;          // phi
    parameters.push_back(param);
    log_prob.push_back(-416.382927743850187661846671194765967569806334854259547205045); // expected log_prob

    param[0] = 100;          // n
    param[1] = -10;          // eta
    param[2] = 200;          // phi
    parameters.push_back(param);
    log_prob.push_back(-1342.30278266569972162264049303129841494915365562553058756128); // expected log_prob
  }

  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // phi
    index.push_back(2U);
    value.push_back(-1);
  }

  template <class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_log_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_log_location& eta, const T_inv_scale& phi,
           const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_2_log_log(n, eta, phi);
  }

  template <bool propto,
            class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_log_location,T_inv_scale>::type
  log_prob(const T_n& n, const T_log_location& eta, const T_inv_scale& phi,
           const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_2_log_log<propto>(n, eta, phi);
  }


  template <class T_n, class T_log_location, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_log_location,T_inv_scale>::type
  log_prob_function(const T_n& n, const T_log_location& eta,
                    const T_inv_scale& phi, const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log_sum_exp;
    using stan::math::multiply_log;

    if (n != 0)
      return binomial_coefficient_log<typename stan::scalar_type<T_inv_scale>::type>(n + phi - 1.0, n)
        +n*eta + multiply_log(phi,phi) - (n+phi)*log_sum_exp(eta,log(phi));
  }
};

