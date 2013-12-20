#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

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
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const char* function = "stan::prob::multi_normal_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_pos_definite;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::dot_product;
      using stan::math::mdivide_left_spd;
      using stan::math::log_determinant;
      
      if (!check_size_match(function, 
                            Sigma.rows(), "Rows of covariance parameter",
                            Sigma.cols(), "columns of covariance parameter",
                            &lp))
        return lp;
      if (!check_positive(function, Sigma.rows(), "Covariance matrix rows", &lp))
        return lp;
      if (!check_symmetric(function, Sigma, "Covariance matrix", &lp))
        return lp;
      if (!check_pos_definite(function, Sigma, "Covariance matrix", &lp))
        return lp;
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
      
      if (include_summand<propto,T_covar>::value) {
        lp -= 0.5 * log_determinant(Sigma);
      }

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_y,T_loc>::type,
            Eigen::Dynamic, 1> y_minus_mu(y.size());
        for (int i = 0; i < y.size(); i++)
          y_minus_mu(i) = y(i)-mu(i);
        Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_covar,T_loc,T_y>::type,
            Eigen::Dynamic, 1> Sinv_y_minus_mu(mdivide_left_spd(Sigma,y_minus_mu));
        lp -= 0.5 * dot_product(y_minus_mu,Sinv_y_minus_mu);
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
      using stan::math::check_pos_definite;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::check_not_nan;
      using stan::math::sum;
      using stan::math::mdivide_left_spd;
      using stan::math::log_determinant;
      using stan::math::columns_dot_product;
      
      if (!check_size_match(function, 
                            Sigma.rows(), "Rows of covariance matrix",
                            Sigma.cols(), "columns of covariance matrix",
                            &lp))
        return lp;
      if (!check_positive(function, Sigma.rows(), "Covariance matrix rows", &lp))
        return lp;
      if (!check_symmetric(function, Sigma, "Covariance matrix", &lp))
        return lp;
      if (!check_pos_definite(function, Sigma, "Covariance matrix", &lp))
        return lp;
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
        lp -= 0.5 * log_determinant(Sigma) * y.rows();
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
        
        // FIXME: revert this code when subtract() is fixed.
        // Eigen::Matrix<typename 
        //               boost::math::tools::promote_args<T_loc,T_y>::type,
        //               Eigen::Dynamic,Eigen::Dynamic> 
        //   z(subtract(y,MU).transpose()); // was = 
        
        Eigen::Matrix<typename 
            boost::math::tools::promote_args<T_covar,T_loc,T_y>::type,
            Eigen::Dynamic,Eigen::Dynamic> Sinv_z(mdivide_left_spd(Sigma,z));
        
        lp -= 0.5 * sum(columns_dot_product(z,Sinv_z));
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
 
      check_positive(function, S.rows(), "Covariance matrix rows");
      check_symmetric(function, S, "Covariance matrix");
      check_finite(function, mu, "Location parameter");

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

