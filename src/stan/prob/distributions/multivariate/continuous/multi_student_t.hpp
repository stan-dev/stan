#ifndef __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__

#include <cstdlib>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/math/error_handling.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>
#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>
#include <stan/math/error_handling/matrix/check_ldlt_factor.hpp>
#include <boost/random/variate_generator.hpp>

namespace stan {

  namespace prob {

    /**
     * Return the log of the multivariate Student t distribution
     * at the specified arguments.
     *
     * @tparam propto Carry out calculations up to a proportion
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_student_t(%1%)";

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_positive;      
      using boost::math::tools::promote_args;
      using boost::math::lgamma;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      using stan::math::check_ldlt_factor;

      typename promote_args<T_y,T_dof,T_loc,T_scale>::type lp(0.0);
      
      // allows infinities
      if (!check_not_nan(function, nu, 
                         "Degrees of freedom parameter", &lp))
        return lp;
      if (!check_positive(function, nu, 
                          "Degrees of freedom parameter", &lp))
        return lp;
      
      using boost::math::isinf;

      if (isinf(nu)) // already checked nu > 0
        return multi_normal_log(y,mu,Sigma);
      
      if (!check_size_match(function, 
          y.size(), "Size of random variable",
          mu.size(), "size of location parameter",
          &lp))
        return lp;
      if (!check_size_match(function, 
          y.size(), "Size of random variable",
          Sigma.rows(), "rows of scale parameter",
          &lp))
        return lp;
      if (!check_size_match(function, 
          y.size(), "Size of random variable",
          Sigma.cols(), "columns of scale parameter",
          &lp))
        return lp;
      if (!check_finite(function, mu, "Location parameter", &lp))
        return lp;
      if (!check_not_nan(function, y, "Random variable", &lp)) 
        return lp;
      if (!check_symmetric(function, Sigma, "Scale parameter", &lp))
        return lp;

      LDLT_factor<T_scale,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      if(!check_ldlt_factor(function,ldlt_Sigma,"LDLT_Factor of scale parameter",&lp))
        return lp;

      double d = y.size();

      if (include_summand<propto,T_dof>::value) {
        lp += lgamma(0.5 * (nu + d));
        lp -= lgamma(0.5 * nu);
        lp -= (0.5 * d) * log(nu);
      }

      if (include_summand<propto>::value) 
        lp -= (0.5 * d) * LOG_PI;

      using stan::math::multiply;
      using stan::math::dot_product;
      using stan::math::subtract;
      using Eigen::Array;


      if (include_summand<propto,T_scale>::value) {
        lp -= 0.5*log_determinant_ldlt(ldlt_Sigma);
      }

      if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
        
        Eigen::Matrix<typename promote_args<T_y,T_loc>::type,
                      Eigen::Dynamic,
                      1> y_minus_mu = subtract(y,mu);
        lp -= 0.5 
          * (nu + d)
          * log(1.0 + trace_inv_quad_form_ldlt(ldlt_Sigma,y_minus_mu) / nu);
      }
      return lp;
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline 
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_student_t_log<false>(y,nu,mu,Sigma);
    }


    template <class RNG>
    inline Eigen::VectorXd
    multi_student_t_rng(const double nu,
                        const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                        const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& s,
                     RNG& rng) {

      static const char* function = "stan::prob::multi_student_t_rng(%1%)";

      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_positive;      
 
      check_finite(function, mu, "Location parameter");
      check_symmetric(function, s, "Scale parameter");
      check_not_nan(function, nu, 
                    "Degrees of freedom parameter");
      check_positive(function, nu, 
                     "Degrees of freedom parameter");

      Eigen::VectorXd z(s.cols());
      z.setZero();
     
      double w = stan::prob::inv_gamma_rng(nu / 2, nu / 2, rng);
      return mu + std::sqrt(w) * stan::prob::multi_normal_rng(z, s, rng);
    }
  }
}
#endif
