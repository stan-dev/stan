// Arguments: Ints, Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

#include <stan/math/functions/lbeta.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBetaBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(4);


    param[0] = 5;            // n
    param[1] = 20;           // N
    param[2] = 10.0;         // alpha
    param[3] = 25.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-1.854007); // expected log_prob

    param[0] = 25;           // n
    param[1] = 100;          // N
    param[2] = 30.0;         // alpha
    param[3] = 50.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-4.376696); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // n
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // alpha
    index.push_back(2U);
    value.push_back(0.0);
    
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(std::numeric_limits<double>::infinity());

    // beta
    index.push_back(3U);
    value.push_back(0.0);
    
    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(std::numeric_limits<double>::infinity());
  }

  template <class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1, T_size2>::type 
  log_prob(const T_n& n, const T_N& N, 
     const T_size1& alpha, const T_size2& beta, 
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::beta_binomial_log(n, N, alpha, beta);
  }

  template <bool propto, 
      class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1, T_size2>::type 
  log_prob(const T_n& n, const T_N& N, 
     const T_size1& alpha, const T_size2& beta, 
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::beta_binomial_log<propto>(n, N, alpha, beta);
  }
  
  
  template <class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  var log_prob_function(const T_n& n, const T_N& N, 
      const T_size1& alpha, const T_size2& beta, 
      const T4&, const T5&, const T6&, 
      const T7&, const T8&, const T9&) {
    using stan::math::lbeta;
    using stan::math::binomial_coefficient_log;
    using stan::prob::include_summand;

    var logp(0);
    if (n < 0 || n > N)
      return logp;
    
    if (include_summand<true>::value)
      logp += binomial_coefficient_log(N,n);
    if (include_summand<true,T_size1,T_size2>::value)
      logp += lbeta(n + alpha, N - n + beta) 
  - lbeta(alpha,beta);
    return logp;
  }
};

TEST(ProbDistributionsBetaBinomialCDF,Values) {
    EXPECT_FLOAT_EQ(0.8868204314, stan::prob::beta_binomial_cdf(49, 100, 1.349, 3.938));
}
