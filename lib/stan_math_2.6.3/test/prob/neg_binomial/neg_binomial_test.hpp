// Arguments: Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/neg_binomial_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsNegBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // alpha
    param[2] = 1.5;          // beta
    parameters.push_back(param);
    log_prob.push_back(-7.786663293475162284096); // expected log_prob

    param[0] = 100;          // n
    param[1] = 3.0;          // alpha
    param[2] = 3.5;          // beta
    parameters.push_back(param);
    log_prob.push_back(-142.6147368129045105434); // expected log_prob

    param[0] = 13;
    param[1] = 1e11; // alpha > 1e10, causes redux to Poisson
    param[2] = 1e10; // equiv to Poisson(1e11/1e10) = Poisson(10)
    parameters.push_back(param);
    log_prob.push_back(-2.6185576442008289933); // log poisson(13|10)

  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // alpha
    index.push_back(1U);
    value.push_back(0);
    
    // beta
    index.push_back(2U);
    value.push_back(0);
  }

  template <class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_shape,T_inv_scale>::type 
  log_prob(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::neg_binomial_log(n, alpha, beta);
  }

  template <bool propto, 
            class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_shape,T_inv_scale>::type 
  log_prob(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
           const T3&, const T4&, const T5) {
    return stan::math::neg_binomial_log<propto>(n, alpha, beta);
  }
  

  template <class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_shape,T_inv_scale>::type 
  log_prob_function(const T_n& n, const T_shape& alpha, 
                        const T_inv_scale& beta,
                        const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;

    if (alpha > 1e10)
        return -lgamma(n + 1.0) + multiply_log(n, alpha / beta) - alpha / beta;
    if (n != 0)
        return binomial_coefficient_log<typename stan::scalar_type<T_shape>::type>(n + alpha - 1.0, n)
          -n * log1p(beta) + alpha * log(beta / (1 + beta));
  }
};

