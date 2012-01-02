#ifndef __STAN__PROB__DISTRIBUTIONS__DIRICHLET_HPP__
#define __STAN__PROB__DISTRIBUTIONS__DIRICHLET_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;
    using stan::maths::multiply_log;

    using Eigen::Dynamic;
    using Eigen::Matrix;

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
    template <bool propto = false,
	      typename T_prob, typename T_prior_sample_size, 
	      class Policy = policy<> > 
    inline typename promote_args<T_prob,T_prior_sample_size>::type
    dirichlet_log(const Matrix<T_prob,Dynamic,1>& theta,
		  const Matrix<T_prior_sample_size,Dynamic,1>& alpha,
		  const Policy& = Policy()) {
      // FIXME: parameter check
      typename promote_args<T_prob,T_prior_sample_size>::type lp(0.0);
      if (include_summand<propto,T_prior_sample_size>::value) {
	lp += lgamma(alpha.sum());
	for (int k = 0; k < alpha.rows(); ++k)
	  lp -= lgamma(alpha[k]);
      }
      if (include_summand<propto,T_prob,T_prior_sample_size>::value)
	for (int k = 0; k < theta.rows(); ++k) 
	  lp += multiply_log(alpha[k]-1, theta[k]);
      return lp;
    }

  }
}
#endif
