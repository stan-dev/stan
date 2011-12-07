#ifndef __STAN__PROB__DISTRIBUTIONS_HPP__
#define __STAN__PROB__DISTRIBUTIONS_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "stan/maths/special_functions.hpp"

#include "stan/prob/transform.hpp"
#include "stan/agrad/matrix.hpp"

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"
#include "stan/prob/distributions_uniform.hpp"
#include "stan/prob/distributions_normal.hpp"
#include "stan/prob/distributions_multi_normal.hpp"
#include "stan/prob/distributions_gamma.hpp"
#include "stan/prob/distributions_inv_gamma.hpp"
#include "stan/prob/distributions_chi_square.hpp"
#include "stan/prob/distributions_inv_chi_square.hpp"
#include "stan/prob/distributions_scaled_inv_chi_square.hpp"
#include "stan/prob/distributions_exponential.hpp"
#include "stan/prob/distributions_wishart.hpp"
#include "stan/prob/distributions_inv_wishart.hpp"
#include "stan/prob/distributions_student_t.hpp"
#include "stan/prob/distributions_beta.hpp"
#include "stan/prob/distributions_dirichlet.hpp"

#include "stan/prob/distributions_cauchy.hpp"
#include "stan/prob/distributions_pareto.hpp"
#include "stan/prob/distributions_double_exponential.hpp"
#include "stan/prob/distributions_weibull.hpp"
#include "stan/prob/distributions_logistic.hpp"
#include "stan/prob/distributions_lognormal.hpp"
#include "stan/prob/distributions_lkj_corr.hpp"
#include "stan/prob/distributions_lkj_cov.hpp"
#include "stan/prob/distributions_bernoulli.hpp"
#include "stan/prob/distributions_categorical.hpp"
/*#include "stan/prob/distributions_binomial.hpp"
#include "stan/prob/distributions_poisson.hpp"
#include "stan/prob/distributions_neg_binomial.hpp"
#include "stan/prob/distributions_beta_binomial.hpp"
#include "stan/prob/distributions_hypergeometric.hpp"
#include "stan/prob/distributions_multinomial.hpp"*/

namespace stan {
  namespace prob {

    using Eigen::Array;
    using Eigen::Matrix;
    using Eigen::DiagonalMatrix;
    using Eigen::Dynamic;
    using namespace std;
    using namespace stan::maths;

    // UNIVARIATE CUMULATIVE DISTRIBUTIONS
    /**
     * Calculates the normal cumulative distribution function for the given
     * y, mean, and variance.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * @param y A scalar variable.
     * @param mean The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified argument.
     * @throw std::domain_error if sigma is less than 0
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale>::type
    normal_p(const T_y& y, const T_loc& mean, const T_scale& sigma) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      return 0.5 * erfc(-(y - mean)/(sigma * SQRT_2));
    }



    // NormalTruncatedLH(y|mu,sigma,low,high)  [sigma > 0, low < high]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (high <= low) {
	std::ostringstream err;
	err << "lower bound (" << low << ") must be less than the upper bound (" << high <<")";
	BOOST_THROW_EXCEPTION (std::invalid_argument (err.str()));
      }

      if (y > high || y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma) - normal_p(low,mu,sigma));
    }

    /**
     * The log of a distribution proportional to a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (high <= low) {
	std::ostringstream err;
	err << "lower bound (" << low << ") must be less than the upper bound (" << high <<")";
	BOOST_THROW_EXCEPTION (std::invalid_argument (err.str()));
      }
      return (normal_trunc_lh_log (y, mu, sigma, low, high));
    }

    // NormalTruncatedL(y|mu,sigma,low)  [sigma > 0]
    // Norm(y|mu,sigma) / (1 - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log1m(normal_p(low,mu,sigma));
    }
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      return normal_trunc_l_log (y, mu, sigma, low);
    }
    // NormalTruncatedH(y|mu,sigma,high)  [sigma > 0]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - 0)
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (y > high)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma));
    }
    /**
     * The log of a density proportional to a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high) {
      if (sigma <= 0) {
	std::ostringstream err;
	err << "sigma (" << sigma << ") must be greater than 0.";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      return normal_trunc_h_log (y, mu, sigma, high);
    }


    // CONTINUOUS, MULTIVARIATE

    

    // DISCRETE, UNIVARIATE MASS FUNCTIONS

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <typename T_n, typename T_N, typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(T_n n, T_N N, T_prob theta) {
      return maths::binomial_coefficient_log<T_N>(N,n)
	+ n * log(theta)
	+ (N - n) * log(1.0 - theta);
    }

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <typename T_rate>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(unsigned int n, T_rate lambda) {
      return - lgamma(n + 1.0)
	+ n * log(lambda)
	- lambda;
    }

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(int n, T_shape alpha, T_inv_scale beta) {
      return maths::binomial_coefficient_log<T_shape>(n + alpha - 1.0, n)
	+ alpha * log(beta / (beta + 1.0))
	+ n * -log(beta + 1.0);
    }

    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <typename T_size>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_log(int n, int N, T_size alpha, T_size beta) {
      return maths::binomial_coefficient_log(N,n)
	+ maths::beta_log(n + alpha, N - n + beta)
	- maths::beta_log(alpha,beta);
    }

    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  a: #white balls;  b: #black balls
    double
    hypergeometric_log(unsigned int n, unsigned int N, 
		       unsigned int a, unsigned int b) {
      return maths::binomial_coefficient_log(a,n)
	+ maths::binomial_coefficient_log(b,N-n)
	- maths::binomial_coefficient_log(a+b,N);
    }

    // DISCRETE, MULTIVARIATE MASS FUNCTIONS

    // Multinomial(ns|N,theta)   [0 <= n <= N;  SUM ns = N;   
    //                            0 <= theta[n] <= 1;  SUM theta = 1]
    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    multinomial_log(std::vector<int>& ns,
		    Matrix<T_prob,Dynamic,1>& theta) {
      unsigned int len = ns.size();
      double sum = 1.0;
      for (unsigned int i = 0; i < len; ++i) 
	sum += ns[i];
      typename boost::math::tools::promote_args<T_prob>::type log_p
	= lgamma(sum);
      for (unsigned int i = 0; i < len; ++i)
	log_p -= lgamma(ns[i] + 1.0);
      for (unsigned int i = 0; i < len; ++i)
	log_p += ns[i] * log(theta[i]);
      return log_p;
    }

    // LINEAR SCALE DENSITIES AND MASS FUNCTIONS
  
    double uniform(double y, double alpha, double beta) {
      return 1.0/(beta - alpha);
    }

    double normal(double y, double loc, double scale) {
      return exp(normal_log(y,loc,scale));
    }

    double gamma(double y, double alpha, double beta) {
      return exp(gamma_log(y,alpha,beta));
    }

    double chi_square(double y, double dof) {
      return exp(chi_square_log(y,dof));
    }

    double inv_chi_square(double y, double dof) {
      return exp(inv_chi_square_log(y,dof));
    }

    double scaled_inv_chi_square(double y, double nu, double s) {
      return exp(scaled_inv_chi_square_log(y,nu,s));
    }

    double exponential(double y, double beta) {
      return exp(exponential_log(y,beta));
    }

  }
}

#endif

