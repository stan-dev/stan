#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP

#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/columns_dot_self.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/agrad/rev/matrix/trace_quad_form.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/error_handling/matrix/check_ldlt_factor.hpp>

namespace stan {

  namespace prob {

    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_prec_log(const T_y& y,
                          const T_loc& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_prec_log(%1%)";
      typedef typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type lp_type;
      lp_type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::sum;
      using stan::math::trace_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      using stan::math::check_ldlt_factor;
      
      check_size_match(function, 
                       Sigma.rows(), "Rows of precision parameter",
                       Sigma.cols(), "columns of precision parameter",
                       &lp);
      check_positive(function, Sigma.rows(), "Precision matrix rows", &lp);
      check_symmetric(function, Sigma, "Precision matrix", &lp);
      
      LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,
                        "LDLT_Factor of precision parameter",&lp);

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
                            Sigma.rows(), "rows of covariance parameter",
                            &lp);
      check_size_match(function, 
                            size_y, "Size of random variable",
                            Sigma.cols(), "columns of covariance parameter",
                            &lp);
  
      for (size_t i = 0; i < size_vec; i++) {      
        check_finite(function, mu_vec[i], "Location parameter", &lp);
        check_not_nan(function, y_vec[i], "Random variable", &lp);
      } 
      
      if (size_y == 0) //y_vec[0].size() == 0
        return lp;
      
      if (include_summand<propto,T_covar>::value)
        lp += 0.5 * log_determinant_ldlt(ldlt_Sigma) * size_vec;

      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * size_y * size_vec;

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Matrix<typename 
              boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type>::type,
              Dynamic, 1> y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          sum_lp_vec += trace_quad_form(Sigma,y_minus_mu);
        }
        lp -= 0.5*sum_lp_vec;
      }
      return lp;
    }
    
    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_prec_log(const T_y& y,
                          const T_loc& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_prec_log<false>(y,mu,Sigma);
    }

  }
}
#endif
  
