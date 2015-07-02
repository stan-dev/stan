// Arguments: Doubles, Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/wiener_log.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <limits>
#include <vector>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionWiener : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(5);

    // using R: RWiener::dwiener(1.1, 2.1, .3, .55, .4, give_log=TRUE)
    param[0] = 1.1;           // y
    param[1] = 2.1;           // alpha
    param[2] = .3;            // tau
    param[3] = .55;           // beta
    param[4] = .4;            // delta
    parameters.push_back(param);
    log_prob.push_back(-0.892976431870503);  // expected log_prob

    param[0] = 2.1;           // y
    param[1] = 4.1;           // alpha
    param[2] = 1.9;           // tau
    param[3] = .05;           // beta
    param[4] = .1;            // delta
    parameters.push_back(param);
    log_prob.push_back(-34.6843978137965);  // expected log_prob

    param[0] = 0.1;           // y
    param[1] = 10.1;          // alpha
    param[2] = 0.05;          // tau
    param[3] = .95;           // beta
    param[4] = .5;            // delta
    parameters.push_back(param);
    log_prob.push_back(0.587463027419532);  // expected log_prob

    param[0] = 50.1;          // y
    param[1] = 4.3;           // alpha
    param[2] = 10.05;         // tau
    param[3] = .65;           // beta
    param[4] = .15;           // delta
    parameters.push_back(param);
    log_prob.push_back(-12.8016739027639);  // expected log_prob

    param[0] = 0.51;          // y
    param[1] = 1.1;           // alpha
    param[2] = .3;            // tau
    param[3] = .2;            // beta
    param[4] = 10.5;          // delta
    parameters.push_back(param);
    log_prob.push_back(-3.04799507974442);  // expected log_prob
  }

  void invalid_values(vector<size_t>& index,
          vector<double>& value) {
    // alpha
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    // tau
    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    // beta
    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

    // delta
    index.push_back(4U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_alpha, typename T_tau,
      typename T_beta, typename T_delta, typename T5>
  typename stan::return_type<T_y, T_alpha, T_tau, T_beta, T_delta>::type
  log_prob(const T_y& y, const T_alpha& alpha, const T_tau& tau,
           const T_beta& beta, const T_delta& delta, const T5&) {
    return stan::math::wiener_log(y, alpha, tau, beta, delta);
  }

  template <bool propto,
      typename T_y, typename T_alpha, typename T_tau,
      typename T_beta, typename T_delta, typename T5>
  typename stan::return_type<T_y, T_alpha, T_tau, T_beta, T_delta>::type
  log_prob(const T_y& y, const T_alpha& alpha, const T_tau& tau,
           const T_beta& beta, const T_delta& delta, const T5&) {
    return stan::math::wiener_log<propto>(y, alpha, tau, beta, delta);
  }


  template <typename T_y, typename T_alpha, typename T_tau,
      typename T_beta, typename T_delta, typename T5>
  typename stan::return_type<T_y, T_alpha, T_tau, T_beta, T_delta, T5>::type
  log_prob_function(const T_y& y, const T_alpha& alpha, const T_tau& tau,
           const T_beta& beta, const T_delta& delta, const T5&) {
    using stan::math::include_summand;
    using stan::math::pi;
    using stan::math::square;
    using boost::math::tools::promote_args;

    // FIXME
    return stan::math::wiener_log<true>(y, alpha, tau, beta, delta);
  }
};

