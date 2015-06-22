// Arguments: Ints, Ints, Doubles
#include <stan/math/prim/scal/prob/binomial_log.hpp>
#include <stan/math/prim/scal/prob/binomial_cdf.hpp>
#include <stan/math/prim/scal/prob/bernoulli_cdf.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = 0.4;          // theta
    parameters.push_back(param);
    log_prob.push_back(-2.144372241799002765106); // expected log_prob
    
    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = 0.8;          // theta
    parameters.push_back(param);
    log_prob.push_back(-9.202729812928724939525); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
   
   
    // N
    index.push_back(1U);
    value.push_back(-1);
   
    // theta
    index.push_back(2U);
    value.push_back(-1e-15);
   
    index.push_back(2U);
    value.push_back(1.0+1e-15);
  }

  template <class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob(const T_n& n, const T_N& N, const T_prob& theta,
           const T3&, const T4&, const T5&) {
    return stan::math::binomial_log(n, N, theta);
  }

  template <bool propto, 
            class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob(const T_n& n, const T_N& N, const T_prob& theta,
           const T3&, const T4&, const T5&) {
    return stan::math::binomial_log<propto>(n, N, theta);
  }
  
  
  template <class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob_function(const T_n& n, const T_N& N, const T_prob& theta,
                    const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;

    return binomial_coefficient_log(N,n) + multiply_log(n,theta) 
      + (N - n) * log1m(theta);
  }
};

TEST(ProbDistributionsBinomialCDF,Values) {
    EXPECT_FLOAT_EQ(0.042817421, stan::math::binomial_cdf(24, 54, 0.57));
    EXPECT_FLOAT_EQ(1.0 - 0.57, stan::math::binomial_cdf(0, 1, 0.57)); // Consistency with expected Bernoulli
    EXPECT_FLOAT_EQ(1.0, stan::math::binomial_cdf(1, 1, 0.57));        // Consistency with expected Bernoulli
    EXPECT_FLOAT_EQ(stan::math::bernoulli_cdf(0, 0.57), stan::math::binomial_cdf(0, 1, 0.57)); // Consistency with implemented Bernoulli
    EXPECT_FLOAT_EQ(stan::math::bernoulli_cdf(1, 0.57), stan::math::binomial_cdf(1, 1, 0.57)); // Consistency with implemented Bernoulli
}
