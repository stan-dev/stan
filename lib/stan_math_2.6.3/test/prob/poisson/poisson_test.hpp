// Arguments: Ints, Doubles
#include <stan/math/prim/scal/prob/poisson_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsPoisson : public AgradDistributionTest {
public:
 void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = 13.0;         // lambda
    parameters.push_back(param);
    log_prob.push_back(-2.900934373290765311282); // expected log_prob

    param[0] = 192;          // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);
    log_prob.push_back(-145.3546649655311853166); // expected log_prob

    param[0] = 0;            // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);
    log_prob.push_back(-3.0); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }

  template <class T_n, class T_rate, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_rate>::type 
  log_prob(const T_n& n, const T_rate& lambda, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::poisson_log(n, lambda);
  }

  template <bool propto, 
            class T_n, class T_rate, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_rate>::type 
  log_prob(const T_n& n, const T_rate& lambda, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::poisson_log<propto>(n, lambda);
  }
  
  
  template <class T_n, class T_rate, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_rate>::type 
  log_prob_function(const T_n& n, const T_rate& lambda, const T2&,
                    const T3&, const T4&, const T5&) {
    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::math::LOG_ZERO;

    if (lambda == 0)
      return n == 0 ? 0 : LOG_ZERO;
    
    if (boost::math::isinf(lambda))
      return LOG_ZERO;
    
    return -lgamma(n + 1.0) + multiply_log(n, lambda) - lambda;
  }
};

