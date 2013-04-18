// Arguments: Double, Double, Double, Double, Double
#include <stan/prob/distributions/univariate/continuous/trunc_normal.hpp>

#include <stan/math/functions/Phi.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionTruncNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(5);

    const double Inf = std::numeric_limits<double>::infinity();

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = -Inf;        // alpha
    param[4] = Inf;         // beta
    parameters.push_back(param);
    log_prob.push_back(-0.9189385); // expected log_prob

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = -Inf;        // alpha
    param[4] = Inf;         // beta
    parameters.push_back(param);
    log_prob.push_back(-1.418939); // expected log_prob

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = -Inf;        // alpha
    param[4] = Inf;         // beta
    parameters.push_back(param);
    log_prob.push_back(-2.918939); // expected log_prob

    param[0] = -3.5;        // y
    param[1] = 1.9;         // mu
    param[2] = 7.2;         // sigma
    param[3] = -Inf;        // alpha
    param[4] = Inf;         // beta
    parameters.push_back(param);
    log_prob.push_back(-3.174270); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T_alpha, typename T_beta, 
      typename T5, typename T6, typename T7, 
      typename T8, typename T9>
  typename stan::return_type<T_y,T_loc,T_scale,T_alpha,T_beta>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_alpha& alpha, const T_beta& beta, 
     const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::trunc_normal_log(y, mu, sigma, alpha, beta);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T_alpha, typename T_beta, 
      typename T5, typename T6, typename T7, 
      typename T8, typename T9>
  typename stan::return_type<T_y,T_loc,T_scale,T_alpha,T_beta>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_alpha& alpha, const T_beta& beta, 
     const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::trunc_normal_log<propto>(y, mu, sigma, alpha, beta);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T_alpha, typename T_beta, 
      typename T5, typename T6, typename T7, 
      typename T8, typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_alpha& alpha, const T_beta& beta, 
      const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::math::check_greater;
    using stan::math::check_not_nan;
    using boost::math::tools::promote_args;
    using boost::math::isinf;
    using boost::math::isfinite;
    using stan::math::Phi;
    using stan::prob::include_summand;
    using stan::prob::normal_log;
      
    var lp(0.0);

    if (y < alpha || y > beta) {
      lp = stan::prob::LOG_ZERO;
    }
    else {
      lp = normal_log<true>(y,mu,sigma);
      if (include_summand<true,T_loc,T_scale,T_alpha,T_beta>::value) {
  if (isinf(sigma)) 
    lp -= log(beta - alpha);
  else {
    if (!isinf(beta) && !isinf(alpha)) 
      lp -= log(Phi((beta - mu)/sigma) - Phi((alpha - mu)/sigma));
    else if (isfinite(alpha)) 
      lp -= log(1.0 - Phi((alpha - mu)/sigma));
    else if (isfinite(beta)) 
      lp -= log(Phi((beta - mu)/sigma));

  }
      }
    }
    return lp;
  }
};

