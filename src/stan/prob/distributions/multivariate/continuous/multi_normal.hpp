#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/check_finite.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>

namespace stan {

  namespace prob {

   template <bool propto,
             typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::check_ldlt_factor;
      
      check_size_match(function, 
                       Sigma.rows(), "Rows of covariance parameter",
                       Sigma.cols(), "columns of covariance parameter",
                       &lp);
      check_positive(function, Sigma.rows(), "Covariance matrix rows", &lp);
      check_symmetric(function, Sigma, "Covariance matrix", &lp);
      
      stan::math::LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,
                        "LDLT_Factor of covariance parameter",&lp);

      check_size_match(function, 
                       y.size(), "Size of random variable",
                       mu.size(), "size of location parameter",
                       &lp);
      check_size_match(function, 
                       y.size(), "Size of random variable",
                       Sigma.rows(), "rows of covariance parameter",
                       &lp);
      check_size_match(function, 
                       y.size(), "Size of random variable",
                       Sigma.cols(), "columns of covariance parameter",
                       &lp);
      check_finite(function, mu, "Location parameter", &lp);
      check_not_nan(function, y, "Random variable", &lp);
      
      if (y.rows() == 0)
        return lp;
      
      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      
      if (include_summand<propto,T_covar>::value) {
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma);
      }

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_y,T_loc>::type,
            Eigen::Dynamic, 1> y_minus_mu(y.size());
        for (int i = 0; i < y.size(); i++)
          y_minus_mu(i) = y(i)-mu(i);
        lp -= 0.5 * trace_inv_quad_form_ldlt(ldlt_Sigma,y_minus_mu);
      }
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_log<false>(y,mu,Sigma);
    }


 

    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::check_not_nan;
      using stan::math::check_ldlt_factor;
      
      check_size_match(function, 
                       Sigma.rows(), "Rows of covariance matrix",
                       Sigma.cols(), "columns of covariance matrix",
                       &lp);
      check_positive(function, Sigma.rows(), "Covariance matrix rows", &lp);
      check_symmetric(function, Sigma, "Covariance matrix", &lp);

      stan::math::LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,"LDLT_Factor of Sigma",&lp);
      
      check_size_match(function, 
                       y.cols(), "Columns of random variable",
                       mu.rows(), "rows of location parameter",
                       &lp);
      check_size_match(function, 
                       y.cols(), "Columns of random variable",
                       Sigma.rows(), "rows of covariance parameter",
                       &lp);
      check_size_match(function, 
                       y.cols(), "Columns of random variable",
                       Sigma.cols(), "columns of covariance parameter",
                       &lp);
      check_finite(function, mu, "Location parameter", &lp);
      check_not_nan(function, y, "Random variable", &lp);
      
      if (y.cols() == 0)
        return lp;
      
      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.cols() * y.rows();
      
      if (include_summand<propto,T_covar>::value) {
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma) * y.rows();
      }
      
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic> MU(y.rows(),y.cols());
        for(typename Eigen::Matrix<T_loc, Eigen::Dynamic, Eigen::Dynamic>::size_type i = 0; 
            i < y.rows(); 
            i++)
          MU.row(i) = mu;
        
        Eigen::Matrix<typename
            boost::math::tools::promote_args<T_loc,T_y>::type,
            Eigen::Dynamic,Eigen::Dynamic> y_minus_MU(y.rows(), y.cols());

        for (int i = 0; i < y.size(); i++)
          y_minus_MU(i) = y(i)-MU(i);
        
        Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_loc,T_y>::type,
            Eigen::Dynamic,Eigen::Dynamic> z(y_minus_MU.transpose()); // was = 
        
        // Eigen::Matrix<typename 
        //               boost::math::tools::promote_args<T_loc,T_y>::type,
        //               Eigen::Dynamic,Eigen::Dynamic> 
        //   z(subtract(y,MU).transpose()); // was = 
        
        lp -= 0.5 * trace_inv_quad_form_ldlt(ldlt_Sigma,z);
      }
      return lp;      
    }

    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_normal_log<false>(y,mu,Sigma);
    }
    
 
    template <class RNG>
    inline Eigen::VectorXd
    multi_normal_rng(const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& S,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;

      static const char* function = "stan::prob::multi_normal_rng(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
 
      check_positive(function, S.rows(), "Covariance matrix rows", (double*)0);
      check_symmetric(function, S, "Covariance matrix", (double*)0);
      check_finite(function, mu, "Location parameter", (double*)0);

      variate_generator<RNG&, normal_distribution<> >
        std_normal_rng(rng, normal_distribution<>(0,1));

      Eigen::VectorXd z(S.cols());
      for(int i = 0; i < S.cols(); i++)
        z(i) = std_normal_rng();

      return mu + S.llt().matrixL() * z;
    }
  }
}

#endif

