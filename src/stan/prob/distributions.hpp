#ifndef __STAN__PROB__DISTRIBUTIONS_HPP__
#define __STAN__PROB__DISTRIBUTIONS_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/throw_exception.hpp>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "stan/maths/special_functions.hpp"
#include "stan/agrad/matrix.hpp"


namespace stan {

  namespace prob {

    using Eigen::Array;
    using Eigen::Matrix;
    using Eigen::DiagonalMatrix;
    using Eigen::Dynamic;
    using namespace std;
    using namespace stan::maths;

    namespace {
   
      const double PI = boost::math::constants::pi<double>();

      const double LOG_ZERO = log(0.0);

      const double LOG_TWO = log(2.0);

      const double NEG_LOG_TWO = -LOG_TWO;

      const double NEG_LOG_SQRT_TWO_PI = - log(sqrt(2.0 * PI));

      const double NEG_LOG_PI = -log(PI);

      const double NEG_LOG_SQRT_PI = -log(sqrt(PI));

      const double NEG_LOG_TWO_OVER_TWO = -LOG_TWO / 2.0;
    }

    // UNIVARIATE CUMULATIVE DISTRIBUTIONS

    namespace {
      const double SQRT_2 = std::sqrt(2);
    }

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
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale>::type
    normal_p(const T_y& y, const T_loc& mean, const T_scale& sigma) {
      return 0.5 * erfc(-(y - mean)/(sigma * SQRT_2));
    }

    // CONTINUOUS, UNIVARIATE DENSITIES

    // Normal(y|mu,sigma)   [sigma > 0]
    /**
     * The log of the normal density for the given y, mean, and standard deviation.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      if (sigma <= 0.0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      return NEG_LOG_SQRT_TWO_PI
	- log(sigma)
	- ((y - mu) * (y - mu)) / (2.0 * sigma * sigma);
    }

    /**
     * The log of the normal density up to a proportion for the given 
     * y, mean, and standard deviation.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density up to a proportion.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */    
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      if (sigma <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      return normal_log(y,mu,sigma);
    }
    
    /**
     * The log of the normal density up to a proportion for the given 
     * y, mean, and standard deviation.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline void
    normal_propto_log(stan::agrad::var& lp, const T_y& y, const T_loc& mu, const T_scale& sigma) {
      if (sigma <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      lp += normal_log(y,mu,sigma);
    }

    // NormalTruncatedLH(y|mu,sigma,low,high)  [sigma > 0, low < high]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, and standard deviation, low, and high.
     * The standard deviation must be greater than 0.
     * high must be greater than low.
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
      if (high < low)
	throw std::invalid_argument ("high must be greater than low");
      if (sigma <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      if (y > high || y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma) - normal_p(low,mu,sigma));
    }

    // NormalTruncatedL(y|mu,sigma,low)  [sigma > 0]
    // Norm(y|mu,sigma) / (1 - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, and standard deviation, the lower bound.
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
    normal_trunc_l_log(T_y y, T_loc mu, T_scale sigma, T_low low) {
      if (sigma <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      if (y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log1m(normal_p(low,mu,sigma));
    }

    // NormalTruncatedH(y|mu,sigma,high)  [sigma > 0]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - 0)
    /**
     * The log of a truncated normal density for the given 
     * y, mean, and standard deviation, the upper bound.
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
    normal_trunc_h_log(T_y y, T_loc mu, T_scale sigma, T_high high) {
      if (sigma <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("sigma must be greater than 0"));
      if (y > high)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma));
    }

    // Uniform(y|alpha,beta)   [alpha < beta;  alpha <= y;  beta <= y]
    /**
     * The log of a uniform density for the given 
     * y, lower, and upper bound.
     * 
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(T_y y, T_low alpha, T_high beta) {
      if (alpha >= beta)
	BOOST_THROW_EXCEPTION(std::invalid_argument ("lower bound must be less than the upper bound"));
      if (y < alpha || y > beta)
	return LOG_ZERO;
      return -log(beta - alpha);
    }


    // Gamma(y|alpha,beta)   [alpha > 0;  beta > 0;  y >= 0]
    /**
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_log(T_y y, T_shape alpha, T_inv_scale beta) {
      if (alpha <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("alpha is <= 0"));
      if (beta <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("beta is <= 0"));
      if (y < 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("y < 0"));
      return - lgamma(alpha)
	+ alpha * log(beta)
	+ (alpha - 1.0) * log(y)
	- beta * y;
    }

    // InvGamma(y|alpha,beta)    [alpha > 0;  beta > 0;  y > 0]
    /**
     * The log of an inverse gamma density for y with the specified
     * shape and inverse scale parameters.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(T_y y, T_shape alpha, T_scale beta) {
      if (alpha <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("alpha is <= 0"));
      if (beta <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("beta is <= 0"));
      if (y <= 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("y <= 0"));
      return - lgamma(alpha)
	+ alpha * log(beta)
	- (alpha + 1) * log(y)
	- beta / y;
    }

    // ChiSquare(y|nu)  [nu >= 0;  y >= 0]
    /**
     * The log of a chi-squared density for y with the specified
     * degrees of freedom parameter.
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <typename T_y, typename T_dof>
    inline typename boost::math::tools::promote_args<T_y,T_dof>::type
    chi_square_log(T_y y, T_dof nu) {
      if (nu < 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("nu is < 0"));
      if (y < 0)
	BOOST_THROW_EXCEPTION(std::domain_error ("y < 0"));
      return - lgamma(0.5 * nu)
	+ nu * NEG_LOG_TWO_OVER_TWO
	+ (0.5 * nu - 1.0) * log(y)
	- 0.5 * y;
    }
  
    // InvChiSquare(y|nu)  [nu > 0;  y > 0]
    template <typename T_y, typename T_dof>
    inline typename boost::math::tools::promote_args<T_y,T_dof>::type
    inv_chi_square_log(T_y y, T_dof nu) {
      return - lgamma(0.5 * nu)
	+ nu * NEG_LOG_TWO_OVER_TWO
	- (0.5 * nu + 1.0) * log(y)
	- 0.5 / y;
    }

    // ScaledInvChiSquare(y|nu,s)  [nu > 0;  s > 0;  y > 0]
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof>::type
    scaled_inv_chi_square_log(T_y y, T_dof nu, T_scale sigma) {
      T_dof half_nu = 0.5 * nu;
      return - lgamma(half_nu)
	+ (half_nu) * log(half_nu)
	+ nu * log(sigma)
	- (half_nu + 1.0) * log(y)
	- half_nu * sigma * sigma / y;
    }

    // Exponential(y|beta) [beta > 0;  y > 0]
    template <typename T_y, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_y,T_inv_scale>::type
    exponential_log(T_y y, T_inv_scale beta) {
      return log(beta)
	- beta * y;
    }
  
    // StudentT(y|nu,mu,sigma)  [nu > 0;   sigma > 0]
    template <typename T_y, 
	      typename T_dof, 
	      typename T_loc, 
	      typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(T_y y, T_dof nu, T_loc mu, T_scale sigma) {
      return lgamma((nu + 1.0) / 2.0)
	- lgamma(nu / 2.0)
	- 0.5 * log(nu)
	+ NEG_LOG_SQRT_PI
	- log(sigma)
	- ((nu + 1.0) / 2.0) * log(1.0 + (((y - mu) / sigma) * ((y - mu) / sigma)) / nu);
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
    template <typename T_y, typename T_prior_sample_size>
    inline typename boost::math::tools::promote_args<T_y,T_prior_sample_size>::type
    beta_log(T_y y, T_prior_sample_size alpha, T_prior_sample_size beta) {
      return lgamma(alpha + beta)
	- lgamma(alpha)
	- lgamma(beta)
	+ (alpha - 1.0) * log(y)
	+ (beta - 1.0) * log(1.0 - y);
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
  

    // CONTINUOUS, MULTIIVARIATE

    // Dirichlet(theta|alpha)    [0 <= theta[n] <= 1;  SUM theta = 1;
    //                            0 < alpha[n]]
    template <typename T_prob, typename T_prior_sample_size> 
    inline typename boost::math::tools::promote_args<T_prob,T_prior_sample_size>::type
    dirichlet_log(Matrix<T_prob,Dynamic,1>& theta,
		  Matrix<T_prior_sample_size,Dynamic,1>& alpha) {
      typename boost::math::tools::promote_args<T_prob,T_prior_sample_size>::type log_p
	= lgamma(alpha.sum());
      for (int k = 0; k < alpha.rows(); ++k)
	log_p -= lgamma(alpha[k]);
      for (int k = 0; k < theta.rows(); ++k) 
	log_p += (alpha[k] - 1) * log(theta[k]);
      return log_p;
    }

    // MultiNormal(y|mu,Sigma)   [y.rows() = mu.rows() = Sigma.rows();
    //                            y.cols() = mu.cols() = 0;
    //                            Sigma symmetric, non-negative, definite]
    template <typename T_y, typename T_loc, typename T_covar> 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(Matrix<T_y,Dynamic,1>& y,
		     Matrix<T_loc,Dynamic,1>& mu,
		     Matrix<T_covar,Dynamic,Dynamic>& Sigma) {
      return NEG_LOG_SQRT_TWO_PI * y.rows()
	- 0.5 * log(Sigma.determinant())
	- 0.5 * ((y - mu).transpose() * Sigma.inverse() * (y - mu))(0,0);
    }

    // MultiNormal(y|mu,L)       [y.rows() = mu.rows() = L.rows() = L.cols();
    //                            y.cols() = mu.cols() = 0;
    //                            Sigma = LL' with L a Cholesky factor]
    template <typename T_y, typename T_loc, typename T_covar> 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(Matrix<T_y,Dynamic,1>& y,
		     Matrix<T_loc,Dynamic,1>& mu,
		     Eigen::TriangularView<T_covar,Eigen::Lower>& L) {
      Matrix<T_covar,Dynamic,1> half = L.solveTriangular(Matrix<T_covar,Dynamic,Dynamic>(L.rows(),L.rows()).setOnes()) * (y - mu);
      return NEG_LOG_SQRT_TWO_PI * y.rows() - log(L.diagonal().array().prod()) - 0.5 * half.dot(half);
    }
   
    namespace {
    }

    // Wishart(Sigma|n,Omega)  [Sigma, Omega symmetric, non-neg, definite; 
    //                          Sigma.dims() = Omega.dims();
    //                           n > Sigma.rows() - 1]
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(Matrix<T_y,Dynamic,Dynamic> W,
		T_dof n,
		Matrix<T_scale,Dynamic,Dynamic> S) {
      unsigned int k = W.rows();
      if (n == (k + 1)) {  
	// don't need W.determinant() term if n == k + 1
	return 	n * k * NEG_LOG_TWO_OVER_TWO
	  - (0.5 * n) * log(S.determinant())
	  - lmgamma(k, 0.5 * n)
	  - 0.5 * abs((S.inverse() * W).trace());
      } else {
	return 0.5 * (n - k - 1.0) * log(W.determinant())
	  + n * k * NEG_LOG_TWO_OVER_TWO
	  - (0.5 * n) * log(S.determinant())
	  - lmgamma(k, 0.5 * n)
	  - 0.5 * abs((S.inverse() * W).trace());
      }
    }

    // InvWishart(Sigma|n,Omega)  [W, S symmetric, non-neg, definite; 
    //                             W.dims() = S.dims();
    //                             n > S.rows() - 1]
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(Matrix<T_y,Dynamic,Dynamic> W,
		    T_dof n,
		    Matrix<T_scale,Dynamic,Dynamic> S) {
      unsigned int k = S.rows();
      return 0.5 * n * log(S.determinant())
	- 0.5 * (n + k + 1.0) * log(W.determinant())
	- 0.5 * (S * W.inverse()).trace()
	+  n * k * NEG_LOG_TWO_OVER_TWO
	- lmgamma(k, 0.5 * n);
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
    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(unsigned int n, unsigned int N, T_prob theta) {
      return maths::binomial_coefficient_log<unsigned int>(N,n)
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

