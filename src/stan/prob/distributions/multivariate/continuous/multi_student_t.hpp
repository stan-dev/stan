#ifndef STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP

#include <cstdlib>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/math/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/subtract.hpp>
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
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type,T_dof,typename scalar_type<T_loc>::type,T_scale>::type
    multi_student_t_log(const T_y& y,
                        const T_dof& nu,
                        const T_loc& mu,
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

      typedef typename boost::math::tools::promote_args<typename scalar_type<T_y>::type,T_dof,typename scalar_type<T_loc>::type,T_scale>::type lp_type;
      lp_type lp(0.0);
      
      // allows infinities
      check_not_nan(function, nu, 
                    "Degrees of freedom parameter", &lp);
      check_positive(function, nu, 
                     "Degrees of freedom parameter", &lp);
      
      using boost::math::isinf;

      if (isinf(nu)) // already checked nu > 0
        return multi_normal_log(y,mu,Sigma);

      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      VectorViewMvt<const T_y> y_vec(y);
      VectorViewMvt<const T_loc> mu_vec(mu);
      //size of std::vector of Eigen vectors
      size_t size_vec = max_size_mvt(y, mu);
      
      
      //Check if every vector of the array has the same size
      int size_y = y_vec[0].size();
      int size_mu = mu_vec[0].size();
      if (size_vec > 1) {
        int size_y_old = size_y;
        int size_y_new;
        for (size_t i = 1, size_ = length_mvt(y); i < size_; i++) {
          int size_y_new = y_vec[i].size();
          check_size_match(function, 
                                size_y_new, "Size of one of the vectors of the random variable",
                                size_y_old, "Size of another vector of the random variable",
                                &lp);
          size_y_old = size_y_new;
        }
        int size_mu_old = size_mu;
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          check_size_match(function, 
                                size_mu_new, "Size of one of the vectors of the location variable",
                                size_mu_old, "Size of another vector of the location variable",
                                &lp);
          size_mu_old = size_mu_new;
        }
        (void) size_y_old;
        (void) size_y_new;
        (void) size_mu_old;
        (void) size_mu_new;
      }

      
      check_size_match(function, 
          size_y, "Size of random variable",
          size_mu, "size of location parameter",
          &lp);
      check_size_match(function, 
          size_y, "Size of random variable",
          Sigma.rows(), "rows of scale parameter",
          &lp);
      check_size_match(function, 
          size_y, "Size of random variable",
          Sigma.cols(), "columns of scale parameter",
          &lp);
      
      for (size_t i = 0; i < size_vec; i++) {
        check_finite(function, mu_vec[i], "Location parameter", &lp);
        check_not_nan(function, y_vec[i], "Random variable", &lp);
      }    
      check_symmetric(function, Sigma, "Scale parameter", &lp);

      
      LDLT_factor<T_scale,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,"LDLT_Factor of scale parameter",
                        &lp);

      if (size_y == 0) //y_vec[0].size() == 0
        return lp;

      if (include_summand<propto,T_dof>::value) {
        lp += lgamma(0.5 * (nu + size_y)) * size_vec;
        lp -= lgamma(0.5 * nu) * size_vec;
        lp -= (0.5 * size_y) * log(nu) * size_vec;
      }

      if (include_summand<propto>::value) 
        lp -= (0.5 * size_y) * LOG_PI * size_vec;

      using stan::math::multiply;
      using stan::math::dot_product;
      using stan::math::subtract;
      using Eigen::Array;


      if (include_summand<propto,T_scale>::value) {
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma) * size_vec;
      }

      if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Matrix<typename 
              boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type>::type,
              Dynamic, 1> y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          sum_lp_vec += log(1.0 + trace_inv_quad_form_ldlt(ldlt_Sigma,y_minus_mu) / nu);
        }
        lp -= 0.5 * (nu + size_y) * sum_lp_vec;
      }
      return lp;
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline 
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type,T_dof,typename scalar_type<T_loc>::type,T_scale>::type
    multi_student_t_log(const T_y& y,
                        const T_dof& nu,
                        const T_loc& mu,
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
 
      check_finite(function, mu, "Location parameter", (double*)0);
      check_symmetric(function, s, "Scale parameter", (double*)0);
      check_not_nan(function, nu, 
                    "Degrees of freedom parameter", (double*)0);
      check_positive(function, nu, 
                     "Degrees of freedom parameter", (double*)0);

      Eigen::VectorXd z(s.cols());
      z.setZero();
     
      double w = stan::prob::inv_gamma_rng(nu / 2, nu / 2, rng);
      return mu + std::sqrt(w) * stan::prob::multi_normal_rng(z, s, rng);
    }
  }
}
#endif
