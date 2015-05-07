// Arguments: Ints, Doubles
#include <stan/math/prim/scal/prob/bernoulli_log.hpp>
#include <stan/math/prim/scal/prob/bernoulli_cdf.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stdexcept>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsBernoulli : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
    index.push_back(1U);
    value.push_back(-0.001);

    index.push_back(1U);
    value.push_back(1.001);
  }

  template <class T_n, class T_prob, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::bernoulli_log(n, theta);
  }

  template <bool propto, 
            class T_n, class T_prob, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::bernoulli_log<propto>(n, theta);
  }
  
  
  template <class T_n, class T_prob, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob_function(const T_n& n, const T_prob& theta, const T2&,
                    const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::log1m;
    if (n == 1)
      return log(theta);
    else if (n == 0)
      return log1m(theta);
    throw std::domain_error("n should either be 1 or 0");
  }
};

TEST(ProbDistributionsBernoulliCDF,Values) {
    EXPECT_FLOAT_EQ(1, stan::math::bernoulli_cdf(1, 0.57));
    EXPECT_FLOAT_EQ(1 - 0.57, stan::math::bernoulli_cdf(0, 0.57));
}
