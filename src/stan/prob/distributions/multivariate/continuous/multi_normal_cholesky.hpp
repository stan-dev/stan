#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_CHOLESKY_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_CHOLESKY_HPP

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
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/sum.hpp>
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
      typedef typename boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type, T_covar>::type lp_type;
      lp_type lp(0.0);

      using stan::math::mdivide_left_tri_low;
      using stan::math::dot_self;
      using stan::math::multiply;
      using stan::math::subtract;
      using stan::math::sum;
      
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;

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
                            L.rows(), "rows of covariance parameter",
                            &lp);
      check_size_match(function, 
                            size_y, "Size of random variable",
                            L.cols(), "columns of covariance parameter",
                            &lp);

      for (size_t i = 0; i < size_vec; i++) {
        check_finite(function, mu_vec[i], "Location parameter", &lp);
        check_not_nan(function, y_vec[i], "Random variable", &lp);
      }
      
      if (size_y == 0)
        return lp;

      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * size_y * size_vec;

      if (include_summand<propto,T_covar>::value)
        lp -= L.diagonal().array().log().sum() * size_vec;

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Eigen::Matrix<typename 
              boost::math::tools::promote_args<typename scalar_type<T_y>::type, typename scalar_type<T_loc>::type>::type,
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
          sum_lp_vec += dot_self(half);
        }
        lp -= 0.5*sum_lp_vec;
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

    template <class RNG>
    inline Eigen::VectorXd
    multi_normal_cholesky_rng(const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& S,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;

      static const char* function = "stan::prob::multi_normal_cholesky_rng(%1%)";

      using stan::math::check_finite;
 
      check_finite(function, mu, "Location parameter", (double*)0);

      variate_generator<RNG&, normal_distribution<> >
        std_normal_rng(rng, normal_distribution<>(0,1));

      Eigen::VectorXd z(S.cols());
      for(int i = 0; i < S.cols(); i++)
        z(i) = std_normal_rng();

      return mu + S * z;
    }
  }
}

#endif
