#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_PREC_HPP__

#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/columns_dot_self.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/matrix/ldlt.hpp>
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

    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                          const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_prec_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::sum;
      using stan::math::trace_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      
      if (!check_size_match(function, 
                            Sigma.rows(), "Rows of covariance parameter",
                            Sigma.cols(), "columns of covariance parameter",
                            &lp))
        return lp;
      if (!check_positive(function, Sigma.rows(), "Precision matrix rows", &lp))
        return lp;
      if (!check_symmetric(function, Sigma, "Precision matrix", &lp))
        return lp;
      
      LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      if (!ldlt_Sigma.success()) {
        std::ostringstream message;
        message << "Precision matrix is not positive definite. " 
        << "Sigma[1,1] is %1%.";
        std::string str(message.str());
        stan::math::dom_err(function,Sigma(0,0),"",str.c_str(),"",&lp);
        return lp;
      }

      if (!check_size_match(function, 
                            y.size(), "Size of random variable",
                            mu.size(), "size of location parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.size(), "Size of random variable",
                            Sigma.rows(), "rows of covariance parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.size(), "Size of random variable",
                            Sigma.cols(), "columns of covariance parameter",
                            &lp))
        return lp;
      if (!check_finite(function, mu, "Location parameter", &lp)) 
        return lp;
      if (!check_not_nan(function, y, "Random variable", &lp)) 
        return lp;
      
      if (y.rows() == 0)
        return lp;
      
      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      
      if (include_summand<propto,T_covar>::value)
        lp += 0.5*log_determinant_ldlt(ldlt_Sigma);
      
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<typename 
          boost::math::tools::promote_args<T_y,T_loc>::type,
          Eigen::Dynamic, 1> y_minus_mu(y.size());
        for (int i = 0; i < y.size(); i++)
          y_minus_mu(i) = y(i)-mu(i);
        lp -= 0.5 * trace_quad_form(Sigma,y_minus_mu);
      }
      return lp;
    }
    
    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                          const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_prec_log<false>(y,mu,Sigma);
    }

    /**
     * y can have multiple rows (observations) and columns (on variables)
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                          const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_prec_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::sum;
      using stan::math::trace_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      
      if (!check_size_match(function, 
                            Sigma.rows(), "Rows of covariance matrix",
                            Sigma.cols(), "columns of covariance matrix",
                            &lp))
        return lp;
      if (!check_positive(function, Sigma.rows(), "Precision matrix rows", &lp))
        return lp;
      if (!check_symmetric(function, Sigma, "Precision matrix", &lp))
        return lp;
      
      LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      if (!ldlt_Sigma.success()) {
        std::ostringstream message;
        message << "Precision matrix is not positive definite. " 
        << "Sigma[1,1] is %1%.";
        std::string str(message.str());
        stan::math::dom_err(function,Sigma(0,0),"",str.c_str(),"",&lp);
        return lp;
      }
      
      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            mu.rows(), "rows of location parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            Sigma.rows(), "rows of covariance parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            y.cols(), "Columns of random variable",
                            Sigma.cols(), "columns of covariance parameter",
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
        lp += log_determinant_ldlt(ldlt_Sigma) * (0.5 * y.cols());
      }
      
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic> MU(y.rows(),y.cols());
        for (typename Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic>::size_type i = 0; 
             i < y.rows(); i++)
          MU.row(i) = mu;
        
        Eigen::Matrix<typename
          boost::math::tools::promote_args<T_loc,T_y>::type,
          Eigen::Dynamic,Eigen::Dynamic> y_minus_MU(y.rows(), y.cols());
        
        for (int i = 0; i < y.size(); i++)
          y_minus_MU(i) = y(i)-MU(i);
        
        Eigen::Matrix<typename 
          boost::math::tools::promote_args<T_loc,T_y>::type,
          Eigen::Dynamic,Eigen::Dynamic> z(y_minus_MU.transpose()); // was = 
        
        lp -= 0.5 * trace_quad_form(Sigma,z);
      }
      return lp;      
    }
    
    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                          const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_prec_log<false>(y,mu,Sigma);
    }

  }
}
#endif
  
