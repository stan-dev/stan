#ifndef STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP

#include <cstdlib>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>
#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>
#include <stan/prob/traits.hpp>

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
      static const std::string function("stan::prob::multi_student_t");

      using stan::error_handling::check_size_match;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_symmetric;
      using stan::error_handling::check_positive;      
      using boost::math::tools::promote_args;
      using boost::math::lgamma;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      using stan::error_handling::check_ldlt_factor;

      typedef typename boost::math::tools::promote_args<typename scalar_type<T_y>::type,T_dof,typename scalar_type<T_loc>::type,T_scale>::type lp_type;
      lp_type lp(0.0);
      
      // allows infinities
      check_not_nan(function, "Degrees of freedom parameter", nu);
      check_positive(function, "Degrees of freedom parameter", nu);
      
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
                           "Size of one of the vectors of the random variable", size_y_new,
                           "Size of another vector of the random variable", size_y_old);
          size_y_old = size_y_new;
        }
        int size_mu_old = size_mu;
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          check_size_match(function, 
                           "Size of one of the vectors of the location variable", size_mu_new, 
                           "Size of another vector of the location variable", size_mu_old);
          size_mu_old = size_mu_new;
        }
        (void) size_y_old;
        (void) size_y_new;
        (void) size_mu_old;
        (void) size_mu_new;
      }

      
      check_size_match(function, 
                       "Size of random variable", size_y, 
                       "size of location parameter", size_mu);
      check_size_match(function, 
                       "Size of random variable", size_y,
                       "rows of scale parameter", Sigma.rows());
      check_size_match(function, 
                       "Size of random variable", size_y,
                       "columns of scale parameter", Sigma.cols());
      
      for (size_t i = 0; i < size_vec; i++) {
        check_finite(function, "Location parameter", mu_vec[i]);
        check_not_nan(function, "Random variable", y_vec[i]);
      }    
      check_symmetric(function, "Scale parameter", Sigma);

      
      LDLT_factor<T_scale,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function, "LDLT_Factor of scale parameter", ldlt_Sigma);

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

      static const std::string function("stan::prob::multi_student_t_rng");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_symmetric;
      using stan::error_handling::check_positive;      
 
      check_finite(function, "Location parameter", mu);
      check_symmetric(function, "Scale parameter", s);
      check_not_nan(function, "Degrees of freedom parameter", nu);
      check_positive(function, "Degrees of freedom parameter", nu);

      Eigen::VectorXd z(s.cols());
      z.setZero();
     
      double w = stan::prob::inv_gamma_rng(nu / 2, nu / 2, rng);
      return mu + std::sqrt(w) * stan::prob::multi_normal_rng(z, s, rng);
    }
  }
}
#endif
