#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_CHOLESKY_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_CHOLESKY_HPP__

#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/columns_dot_self.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the multivariate normal density for the given y, mu, and
     * a Cholesky factor L of the variance matrix.
     * Sigma = LL', a square, semi-positive definite matrix.
     *
     *
     * @param y A scalar vector
     * @param mu The mean vector of the multivariate normal distribution.
     * @param L The Cholesky decomposition of a variance matrix 
     * of the multivariate normal distribution
     * @return The log of the multivariate normal density.
     * @throw std::domain_error if LL' is not square, not symmetric, 
     * or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_covar Type of scale.
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_cholesky_log(const T_y& y,
                              const T_loc& mu,
                              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L) {
      static const char* function = "stan::prob::multi_normal_cholesky_log(%1%)";

      using stan::math::mdivide_left_tri_low;
      using stan::math::dot_self;
      using stan::math::multiply;
      using stan::math::subtract;
      using stan::math::sum;
      
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_cov_matrix;
      using boost::math::tools::promote_args;

      typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type lp(0.0);

      VectorViewMvt<const T_y> y_vec(y);
      VectorViewMvt<const T_loc> mu_vec(mu);
      size_t size = max_size_mvt(y, mu);

      //Check if every vector of the array has the same size
      int size_y = y_vec[0].size();
      if (size > 1) {
        int size_y_old = size_y;
        int size_y_new;
        for (size_t i = 1, size_ = length_mvt(y); i < size_; i++) {
          int size_y_new = y_vec[i].size();
          if (!check_size_match(function, 
                                size_y_new, "Size of one of the vectors of the random variable",
                                size_y_old, "Size of another vector of the random variable",
                                &lp))
            return lp;          
          size_y_old = size_y_new;
        }
        int size_mu_old = mu_vec[0].size();
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          if (!check_size_match(function, 
                                size_mu_new, "Size of one of the vectors of the location variable",
                                size_mu_old, "Size of another vector of the location variable",
                                &lp))
            return lp;          
          size_mu_old = size_mu_new;
        }
      }

    
      if (!check_size_match(function, 
                            size_y, "Size of random variable",
                            mu_vec[0].size(), "size of location parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            size_y, "Size of random variable",
                            L.rows(), "rows of covariance parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            size_y, "Size of random variable",
                            L.cols(), "columns of covariance parameter",
                            &lp))
        return lp;
        
      for (size_t i = 0; i < size; i++) { 
        if (!check_finite(function, mu_vec[i], "Location parameter", &lp)) 
          return lp;
        if (!check_not_nan(function, y_vec[i], "Random variable", &lp)) 
          return lp;
      }
      
      if (size_y == 0)
        return lp;

      
        if (include_summand<propto>::value) 
          lp += NEG_LOG_SQRT_TWO_PI * size_y * size;
        
        if (include_summand<propto,T_covar>::value) {
          Eigen::Matrix<T_covar,Eigen::Dynamic,1> L_log_diag = L.diagonal().array().log().matrix();
          lp -= sum(L_log_diag) * size;
        }
        
      for (size_t i = 0; i < size; i++) {      
        if (include_summand<propto,T_y,T_loc,T_covar>::value) {
          Eigen::Matrix<typename 
            boost::math::tools::promote_args<typename scalar_type<T_y>::type,typename scalar_type<T_loc>::type>::type,
            Eigen::Dynamic, 1> y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_covar,typename scalar_type<T_loc>::type,typename scalar_type<T_y>::type>::type,
            Eigen::Dynamic, 1> 
            half(mdivide_left_tri_low(L,y_minus_mu));
          // FIXME: this code does not compile. revert after fixing subtract()
          // Eigen::Matrix<typename 
          //               boost::math::tools::promote_args<T_covar,typename scalar_type<T_loc>::type,typename scalar_type<T_y>::type>::type>::type,
          //               Eigen::Dynamic, 1> 
          //   half(mdivide_left_tri_low(L,subtract(y,mu)));
          lp -= 0.5 * dot_self(half);
        }
      }
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_cholesky_log(const T_y& y,
                              const T_loc& mu,
                              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L) {
      return multi_normal_cholesky_log<false>(y,mu,L);
    }

    /** y can have multiple rows (observations) and columns (on variables)
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_cholesky_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                              const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L) {
      static const char* function = "stan::prob::multi_normal_cholesky_log(%1%)";

      using stan::math::mdivide_left_tri_low;
      using stan::math::columns_dot_self;
      using stan::math::multiply;
      using stan::math::subtract;
      using stan::math::sum;
      using stan::math::log;
      
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_cov_matrix;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_covar>::type lp(0.0);

      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            mu.rows(), "rows of location parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            L.rows(), "rows of covariance parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            L.cols(), "columns of covariance parameter",
                            &lp))
        return lp;
      if (!check_finite(function, mu, "Location parameter", &lp)) 
        return lp;
      if (!check_not_nan(function, y, "Random variable", &lp)) 
        return lp;

      if (y.cols() == 0)
        return lp;

      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.cols() * y.rows();

      if (include_summand<propto,T_covar>::value) {
        Eigen::Matrix<T_covar,Eigen::Dynamic,1> L_log_diag = L.diagonal().array().log().matrix();
        lp -= sum(L_log_diag) * y.rows();
      }

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic> MU(y.rows(),y.cols());
        for (typename Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic>::size_type i = 0; 
             i < y.rows(); i++)
          MU.row(i) = mu;
  
        Eigen::Matrix<typename
          boost::math::tools::promote_args<T_loc,T_y>::type,
          Eigen::Dynamic,Eigen::Dynamic>
          y_minus_MU(y.rows(), y.cols());
        for (int i = 0; i < y.size(); i++)
          y_minus_MU(i) = y(i)-MU(i);

        Eigen::Matrix<typename 
          boost::math::tools::promote_args<T_loc,T_y>::type,
          Eigen::Dynamic,Eigen::Dynamic> 
          z(y_minus_MU.transpose()); // was = 
        
        // FIXME: revert this code when subtract() is fixed.
        // Eigen::Matrix<typename 
        //               boost::math::tools::promote_args<T_loc,T_y>::type,
        //               Eigen::Dynamic,Eigen::Dynamic> 
        //   z(subtract(y,MU).transpose()); // was = 
                
        Eigen::Matrix<typename 
          boost::math::tools::promote_args<T_covar,T_loc,T_y>::type,
          Eigen::Dynamic,Eigen::Dynamic> 
          half(mdivide_left_tri_low(L,z));
          
        lp -= 0.5 * sum(columns_dot_self(half));
      }
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_cholesky_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                              const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L) {
      return multi_normal_cholesky_log<false>(y,mu,L);
    }

  }
}

#endif
