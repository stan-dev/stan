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

#include "stan/prob/distributions_student_t.hpp"

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


    // Cauchy(y|mu,sigma)  [sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(T_y y, T_loc mu, T_scale sigma) {
      return NEG_LOG_PI
	- log(sigma)
	- log(1.0 + (y - mu) * (y - mu) / (sigma * sigma));
    }

    // Beta(y|alpha,beta)  [alpha > 0;  beta > 0;  0 <= y <= 1]
    /**
     * The log of a beta density for y with the specified
     * prior sample sizes.
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * y must be between 0 and 1 inclusive.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Beta}}(\alpha, \beta) \\
       \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} y^{\alpha - 1} (1-y)^{\beta - 1} \right) \\
       &=& \log (\Gamma(\alpha + \beta)) - \log (\Gamma (\alpha) - \log(\Gamma(\beta)) + (\alpha-1) \log(y) + (\beta-1) \log(1 - y) \\
       & & \mathrm{where} \; y \in [0, 1]
     \f}
     * @param y A scalar variable.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_alpha Type of prior sample size for alpha.
     * @tparam T_beta Type of prior sample size for beta.
     */
    template <typename T_y, typename T_alpha, typename T_beta>
    inline typename boost::math::tools::promote_args<T_y,T_alpha,T_beta>::type
    beta_log(const T_y& y, const T_alpha& alpha, const T_beta& beta) {
      return lgamma(alpha + beta)
	- lgamma(alpha)
	- lgamma(beta)
	+ (alpha - 1.0) * log(y)
	+ (beta - 1.0) * log(1.0 - y);
    }
    /**
     * The log of a distribution proportional to a beta density for y with the specified
     * prior sample sizes.
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * y must be between 0 and 1 inclusive.
     *
     * @param y A scalar variable.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_alpha Type of prior sample size for alpha.
     * @tparam T_beta Type of prior sample size for beta.
     */
    template <typename T_y, typename T_alpha, typename T_beta>
    inline typename boost::math::tools::promote_args<T_y,T_alpha,T_beta>::type
    beta_propto_log(const T_y& y, const T_alpha& alpha, const T_beta& beta) {
      return beta_log (y, alpha, beta);
    }

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <typename T_y, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(T_y y, T_scale y_min, T_shape alpha) {
      return log(alpha)
	+ alpha * log(y_min)
	- (alpha + 1.0) * log(y);
    }

    // DoubleExponential(y|mu,sigma)  [sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale> 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(T_y y, T_loc mu, T_scale sigma) {
      return NEG_LOG_TWO
	- log(sigma)
	- abs(y - mu) / sigma;
    }

    // Weibull(y|sigma,alpha)     [y >= 0;  sigma > 0;  alpha > 0]
    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(T_y y, T_shape alpha, T_scale sigma) {
      return log(alpha)
	- log(sigma)
	+ (alpha - 1.0) * (log(y) - log(sigma))
	- pow(y / sigma, alpha);
    }

    // Logistic(y|mu,sigma)    [sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(T_y y, T_loc mu, T_scale sigma) {
      return -(y - mu)/sigma
	- log(sigma)
	- 2.0 * log(1.0 + exp(-(y - mu)/sigma));
    }

    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(T_y y, T_loc mu, T_scale sigma) {
      return NEG_LOG_SQRT_TWO_PI
	- log(sigma)
	- log(y)
	- pow(log(y) - mu,2.0) / (2.0 * sigma * sigma);
    }
  

    // CONTINUOUS, MULTIVARIATE

    // Dirichlet(theta|alpha)    [0 <= theta[n] <= 1;  SUM theta = 1;
    //                            0 < alpha[n]]
    /**
     * The log of the Dirichlet density for the given theta and
     * a vector of prior sample sizes, alpha.
     * Each element of alpha must be greater than 0. 
     * Each element of theta must be greater than or 0.
     * Theta sums to 1.
     *
     * \f{eqnarray*}{
       \theta &\sim& \mbox{\sf{Dirichlet}} (\alpha_1, \ldots, \alpha_k) \\
       \log (p (\theta \,|\, \alpha_1, \ldots, \alpha_k) ) &=& \log \left( \frac{\Gamma(\alpha_1 + \cdots + \alpha_k)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_k)}
          \theta_1^{\alpha_1 - 1} \cdots \theta_k^{\alpha_k - 1} \right) \\
       &=& \log (\Gamma(\alpha_1 + \cdots + \alpha_k)) - \log(\Gamma(\alpha_1)) - \cdots - \log(\Gamma(\alpha_k)) +
           (\alpha_1 - 1) \log (\theta_1) + \cdots + (\alpha_k - 1) \log (\theta_k)
     \f}
     * 
     * @param theta A scalar vector.
     * @param alpha Prior sample sizes.
     * @return The log of the Dirichlet density.
     * @throw std::domain_error if any element of alpha is less than or equal to 0.
     * @throw std::domain_error if any element of theta is less than 0.
     * @throw std::domain_error if the sum of theta is not 1.
     * @tparam T_prob Type of scalar.
     * @tparam T_prior_sample_size Type of prior sample sizes.
     */
    template <typename T_prob, typename T_prior_sample_size> 
    inline typename boost::math::tools::promote_args<T_prob,T_prior_sample_size>::type
    dirichlet_log(const Matrix<T_prob,Dynamic,1>& theta,
		  const Matrix<T_prior_sample_size,Dynamic,1>& alpha) {
      // FIXME: parameter check
      typename boost::math::tools::promote_args<T_prob,T_prior_sample_size>::type log_p
	= lgamma(alpha.sum());
      for (int k = 0; k < alpha.rows(); ++k)
	log_p -= lgamma(alpha[k]);
      for (int k = 0; k < theta.rows(); ++k) 
	log_p += (alpha[k] - 1) * log(theta[k]);
      return log_p;
    }
    /**
     * The log of a density proportional to the Dirichlet density for the given theta and
     * a vector of prior sample sizes, alpha.
     * Each element of alpha must be greater than 0. 
     * Each element of theta must be greater than or 0.
     * Theta sums to 1.
     *
     * @param theta A scalar vector.
     * @param alpha Prior sample sizes.
     * @return The log of the Dirichlet density.
     * @throw std::domain_error if any element of alpha is less than or equal to 0.
     * @throw std::domain_error if any element of theta is less than 0.
     * @throw std::domain_error if the sum of theta is not 1.
     * @tparam T_prob Type of scalar.
     * @tparam T_prior_sample_size Type of prior sample sizes.
     */
    template <typename T_prob, typename T_prior_sample_size> 
    inline typename boost::math::tools::promote_args<T_prob,T_prior_sample_size>::type
    dirichlet_propto_log(const Matrix<T_prob,Dynamic,1>& theta,
			 const Matrix<T_prior_sample_size,Dynamic,1>& alpha) {
      // FIXME: parameter check
      return dirichlet_log (theta, alpha);
    }

    // Wishart(Sigma|n,Omega)  [Sigma, Omega symmetric, non-neg, definite; 
    //                          Sigma.dims() = Omega.dims();
    //                           n > Sigma.rows() - 1]
    /**
     * The log of the Wishart density for the given W, degrees of freedom, 
     * and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * \f{eqnarray*}{
       W &\sim& \mbox{\sf{Wishart}}_{\nu} (S) \\
       \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
                                                  \times \left| S \right|^{-\nu/2} \left| W \right|^{(\nu - k - 1) / 2}
						  \times \exp (-\frac{1}{2} \mathsf{tr} (S^{-1} W)) \right) \\
       &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
           -\frac{\nu}{2} \log(\det(S)) + \frac{\nu-k-1}{2}\log (\det(W)) - \frac{1}{2} \mathsf{tr} (S^{-1}W)
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Matrix<T_y,Dynamic,Dynamic>& W,
		const T_dof& nu,
		const Matrix<T_scale,Dynamic,Dynamic>& S) {
      // FIXME: domain checks
      unsigned int k = W.rows();
      if (nu <= k - 1) {
	std::ostringstream err;
	err << "nu (" << nu << ") must be greater than k-1 (" << k-1 << ")";
	BOOST_THROW_EXCEPTION (std::domain_error(err.str()));
      }
      if (nu == (k + 1)) {  
	// don't need W.determinant() term if n == k + 1
	return 	nu * k * NEG_LOG_TWO_OVER_TWO
	  - (0.5 * nu) * log(S.determinant())
	  - lmgamma(k, 0.5 * nu)
	  - 0.5 * abs((S.inverse() * W).trace());
      } else {
	return 0.5 * (nu - k - 1.0) * log(W.determinant())
	  + nu * k * NEG_LOG_TWO_OVER_TWO
	  - (0.5 * nu) * log(S.determinant())
	  - lmgamma(k, 0.5 * nu)
	  - 0.5 * abs((S.inverse() * W).trace());
      }
    }
    /**
     * The log of a density proportional to a Wishart density for the given W,
     * degrees of freedom, and scale matrix. 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Wishart density at W given nu and S.
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_propto_log(const Matrix<T_y,Dynamic,Dynamic>& W,
		       const T_dof& nu,
		       const Matrix<T_scale,Dynamic,Dynamic>& S) {
      if (nu <= S.rows() - 1) {
	std::ostringstream err;
	err << "nu (" << nu << ") must be greater than k-1 (" << S.rows()-1 << ")";
	BOOST_THROW_EXCEPTION (std::domain_error(err.str()));
      }
      // FIXME: domain checks
      return wishart_log (W, nu, S);
    }

    // InvWishart(Sigma|n,Omega)  [W, S symmetric, non-neg, definite; 
    //                             W.dims() = S.dims();
    //                             n > S.rows() - 1]
    /**
     * The log of the Inverse-Wishart density for the given W, degrees of freedom, 
     * and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * \f{eqnarray*}{
       W &\sim& \mbox{\sf{Inv-Wishart}}_{\nu} (S) \\
       \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
                                                  \times \left| S \right|^{\nu/2} \left| W \right|^{-(\nu + k + 1) / 2}
						  \times \exp (-\frac{1}{2} \mathsf{tr} (S W^{-1})) \right) \\
       &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
           +\frac{\nu}{2} \log(\det(S)) - \frac{\nu+k+1}{2}\log (\det(W)) - \frac{1}{2} \mathsf{tr}(S W^{-1})
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Matrix<T_y,Dynamic,Dynamic>& W,
		    const T_dof& nu,
		    const Matrix<T_scale,Dynamic,Dynamic>& S) {
      unsigned int k = S.rows();
      if (nu <= k - 1) {
	std::ostringstream err;
	err << "nu (" << nu << ") must be greater than k-1 (" << k-1 << ")";
	BOOST_THROW_EXCEPTION (std::domain_error(err.str()));
      }
      return 0.5 * nu * log(S.determinant())
	- 0.5 * (nu + k + 1.0) * log(W.determinant())
	- 0.5 * (S * W.inverse()).trace()
	+  nu * k * NEG_LOG_TWO_OVER_TWO
	- lmgamma(k, 0.5 * nu);
    }
    /**
     * The log of a density proportional to the Inverse-Wishart density for 
     * the given W, degrees of freedom, and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_propto_log(const Matrix<T_y,Dynamic,Dynamic>& W,
			   const T_dof& nu,
			   const Matrix<T_scale,Dynamic,Dynamic>& S) {
      if (nu <= S.rows() - 1) {
	std::ostringstream err;
	err << "nu (" << nu << ") must be greater than k-1 (" << S.rows()-1 << ")";
	BOOST_THROW_EXCEPTION (std::domain_error(err.str()));
      }
      return inv_wishart_log (W, nu, S);
    }
    // ?? write these in terms of cpcs rather than corr matrix
    
    // LKJ_Corr(y|eta) [ y correlation matrix (not covariance matrix)
    //                  eta > 0 ]
    template <typename T_y, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(Matrix<T_y,Dynamic,Dynamic> y,
		 T_shape eta) {

      // Lewandowski, Kurowicka, and Joe (2009) equations 15 and 16
      
      const unsigned int K = y.rows();
      T_shape the_sum = 0.0;
      T_shape constant = 0.0;
      T_shape beta_arg;
      
      if(eta == 1.0) {
	for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	  beta_arg = 0.5 * (k + 1.0);
	  constant += k * beta_log(beta_arg, beta_arg);
	  the_sum += pow(static_cast<double>(k),2.0);
	}
	constant += the_sum * LOG_TWO;
	return constant;
      }

      T_shape diff;
      for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	diff = K - k;
	beta_arg = eta + 0.5 * (diff - 1);
	constant += diff * beta_log(beta_arg, beta_arg);
	the_sum += (2.0 * eta - 2.0 + diff) * diff;
      }
      constant += the_sum * LOG_TWO;
      return (eta - 1.0) * log(y.determinant()) + constant;
    }

    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(Matrix<T_y,Dynamic,Dynamic> y,
		Matrix<T_loc,Dynamic,1> mu,
		Matrix<T_scale,Dynamic,1> sigma,
		T_shape eta) {

      const unsigned int K = y.rows();
      const Array<T_y,Dynamic,1> sds = y.diagonal().array().sqrt();
      T_shape log_prob = 0.0;
      for(unsigned int k = 0; k < K; k++) {
	log_prob += lognormal_log(log(sds(k,1)), mu(k,1), sigma(k,1));
      }
      if(eta == 1.0) {
	// no need to rescale y into a correlation matrix
	log_prob += lkj_corr_log(y,eta); 
	return log_prob;
      }
      DiagonalMatrix<double,Dynamic> D(K);
      D.diagonal() = sds.inverse();
      log_prob += lkj_corr_log(D * y * D, eta);
      return log_prob;
    }

    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(Matrix<T_y,Dynamic,Dynamic> y,
		T_loc mu, T_scale sigma, T_shape eta) {

      const unsigned int K = y.rows();
      const Array<T_y,Dynamic,1> sds = y.diagonal().array().sqrt();
      T_shape log_prob = 0.0;
      for(unsigned int k = 0; k < K; k++) {
	log_prob += lognormal_log(sds(k,1), mu, sigma);
      }
      if(eta == 1.0) {
	log_prob += lkj_corr_log(y,eta); // no need to rescale y into a correlation matrix
	return log_prob;
      }
      DiagonalMatrix<double,Dynamic> D(K);
      D.diagonal() = sds.inverse();
      log_prob += lkj_corr_log(D * y * D, eta);
      return log_prob;
    }

    // DISCRETE, UNIVARIATE MASS FUNCTIONS

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    template <typename T_prob> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(unsigned int n, T_prob theta) {
      return log(n ? theta : (1.0 - theta));
    }

    // Categorical(n|theta)  [0 <= n < N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(unsigned int n, Matrix<T_prob,Dynamic,1>& theta) {
      return log(theta(n));
    }

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

