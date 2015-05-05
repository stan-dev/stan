#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>

#include <stan/math/prim/mat/prob/multi_normal_prec_log.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/is_nan.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/mat/fun/trace_quad_form.hpp>

#include <vector>
#include <test/unit/math/rev/mat/prob/test_gradients.hpp>
#include <test/unit/math/rev/mat/prob/test_gradients_multi_normal.hpp>
#include <test/unit/math/rev/mat/prob/expect_eq_diffs.hpp>
#include <test/unit/math/prim/mat/prob/agrad_distributions_multi_normal_multi_row.hpp>
#include <test/unit/math/prim/mat/prob/agrad_distributions_multi_normal.hpp>


using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

using stan::math::var;
using stan::math::to_var;

template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::math::multi_normal_prec_log<false>(y1,mu1,sigma1),
                  stan::math::multi_normal_prec_log<false>(y2,mu2,sigma2),
                  stan::math::multi_normal_prec_log<true>(y1,mu1,sigma1),
                  stan::math::multi_normal_prec_log<true>(y2,mu2,sigma2),
                  message);
}


TEST_F(agrad_distributions_multi_normal,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(agrad_distributions_multi_normal,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(agrad_distributions_multi_normal,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(agrad_distributions_multi_normal,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(agrad_distributions_multi_normal,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}

TEST_F(agrad_distributions_multi_normal_multi_row,Propto) {
  expect_propto(to_var(y),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(mu2),to_var(Sigma2),
                "All vars: y, mu, sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoY) {
  expect_propto(to_var(y),mu,Sigma,
                to_var(y2),mu,Sigma,
                "var: y");

}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoYMu) {
  expect_propto(to_var(y),to_var(mu),Sigma,
                to_var(y2),to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoYSigma) {
  expect_propto(to_var(y),mu,to_var(Sigma),
                to_var(y2),mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoMu) {
  expect_propto(y,to_var(mu),Sigma,
                y,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoMuSigma) {
  expect_propto(y,to_var(mu),to_var(Sigma),
                y,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(agrad_distributions_multi_normal_multi_row,ProptoSigma) {
  expect_propto(y,mu,to_var(Sigma),
                y,mu,to_var(Sigma2),
                "var: sigma");
}



struct multi_normal_prec_fun {
  const int K_;

  multi_normal_prec_fun(int K) : K_(K) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::math::var;
    Matrix<T,Dynamic,1> y(K_);
    Matrix<T,Dynamic,1> mu(K_);
    Matrix<T,Dynamic,Dynamic> Sigma(K_,K_);
    int pos = 0;
    for (int i = 0; i < K_; ++i)
      y(i) = x[pos++];
    for (int i = 0; i < K_; ++i)
      mu(i) = x[pos++];
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = x[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    return stan::math::multi_normal_prec_log<false>(y,mu,Sigma);
  }
};

TEST(MultiNormalPrec, TestGradFunctional) {
  std::vector<double> x(3 + 3 + 3 * 2);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;
  // mu
  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;
  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;
  x[9] = -2;
  x[10] = 20;
  x[11] = 56;

  test_grad(multi_normal_prec_fun(3), x);

  std::vector<double> u(3);
  u[0] = 1.9;
  u[1] = -2.7;
  u[2] = 0.48;
  
  test_grad(multi_normal_prec_fun(1), u);

}

template <int is_row_vec_y, int is_row_vec_mu>
struct vectorized_multi_normal_prec_fun {
  const int K_; //size of each vector and order of square matrix sigma
  const int L_; //size of the array of eigen vectors
  const bool dont_vectorize_y; //direct use eigen vector for y
  const bool dont_vectorize_mu; //direct use eigen vector for mu
  
  vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(int K, int L, bool M = false, bool N = false) : K_(K), L_(L), 
                                        dont_vectorize_y(M),
                                        dont_vectorize_mu(N) {
    if ((dont_vectorize_y || dont_vectorize_mu) && L != 1)
      throw std::runtime_error("attempt to disable vectorization with vector bigger than 1");
  }

  template <typename T_y, typename T_mu, typename T_sigma>
  typename boost::math::tools::promote_args<T_y, T_mu, T_sigma>::type
  operator() (const std::vector<T_y>& y_vec,
              const std::vector<T_mu>& mu_vec,
              const std::vector<T_sigma>& sigma_vec) const {
    vector<Matrix<T_y,is_row_vec_y,is_row_vec_y*-1> > y(L_, Matrix<T_y,is_row_vec_y,is_row_vec_y*-1> (K_));
    vector<Matrix<T_mu,is_row_vec_mu,is_row_vec_mu*-1> > mu(L_, Matrix<T_mu,is_row_vec_mu,is_row_vec_mu*-1> (K_));
    Matrix<T_sigma,Dynamic,Dynamic> Sigma(K_, K_);
    int pos = 0;
    for (int i = 0; i < L_; ++i) 
      for (int j = 0; j < K_; ++j)
        y[i](j) = y_vec[pos++];

    pos = 0;        
    for (int i = 0; i < L_; ++i)         
      for (int j = 0; j < K_; ++j)
        mu[i](j) = mu_vec[pos++];
    
    pos = 0;
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = sigma_vec[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    
    if (dont_vectorize_y) {
      if (dont_vectorize_mu)
        return stan::math::multi_normal_prec_log<false>(y[0], mu[0], Sigma);
      else
        return stan::math::multi_normal_prec_log<false>(y[0], mu, Sigma);
    }
    else {
      if (dont_vectorize_mu)
        return stan::math::multi_normal_prec_log<false>(y, mu[0], Sigma);
      else
        return stan::math::multi_normal_prec_log<false>(y, mu, Sigma);
    }
  }
};

template <int is_row_vec_y, int is_row_vec_mu>
void test_all() {
  {
    vector<double> y_(3), mu_(3), sigma_(6);
    // y
    y_[0] = 1.0;
    y_[1] = 2.0;
    y_[2] = -3.0;
    // mu
    mu_[0] = 0.0;
    mu_[1] = -2.0;
    mu_[2] = -3.0;
    // Sigma
    sigma_[0] = 1;
    sigma_[1] = -1;
    sigma_[2] = 10;
    sigma_[3] = -2;
    sigma_[4] = 20;
    sigma_[5] = 56;
    for (int ii = 0; ii < 2; ii++)
      for (int jj = 0; jj < 2; jj++) {
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               y_, mu_, sigma_);
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               y_, mu_, get_vvar(sigma_));
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               y_, get_vvar(mu_), sigma_);
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               y_, get_vvar(mu_), get_vvar(sigma_));
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               get_vvar(y_), mu_, sigma_);
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               get_vvar(y_), mu_, get_vvar(sigma_));
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               get_vvar(y_), get_vvar(mu_), sigma_);
        test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 1, ii, jj),
                               get_vvar(y_), get_vvar(mu_), get_vvar(sigma_));
    }
  }
  
  {
    vector<double> y_(6), mu_(6), sigma_(6);
    // y[1]
    y_[0] = 1.0;
    y_[1] = 2.0;
    y_[2] = -3.0;
    // y[2]
    y_[3] = 0.0;
    y_[4] = -2.0;
    y_[5] = -3.0;
    
    // mu[1]
    mu_[0] = 0.0;
    mu_[1] = 1.0;
    mu_[2] = 3.0;
    // mu[2]
    mu_[3] = 0.0;
    mu_[4] = -1.0;
    mu_[5] = -2.0;
    
    // Sigma
    sigma_[0] = 1;
    sigma_[1] = -1;
    sigma_[2] = 10;
    sigma_[3] = -2;
    sigma_[4] = 20;
    sigma_[5] = 56;
 
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           y_, mu_, sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           y_, mu_, get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           y_, get_vvar(mu_), sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           y_, get_vvar(mu_), get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           get_vvar(y_), mu_, sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           get_vvar(y_), mu_, get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           get_vvar(y_), get_vvar(mu_), sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(3, 2),
                           get_vvar(y_), get_vvar(mu_), get_vvar(sigma_));
  }
  {
    vector<double> y_(1), mu_(1), sigma_(1);
    y_[0] = 1.9;
    mu_[0] = -2.7;
    sigma_[0] = 0.48;
    
    
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           y_, mu_, sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           y_, mu_, get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           y_, get_vvar(mu_), sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           y_, get_vvar(mu_), get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           get_vvar(y_), mu_, sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           get_vvar(y_), mu_, get_vvar(sigma_));
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           get_vvar(y_), get_vvar(mu_), sigma_);
    test_grad_multi_normal(vectorized_multi_normal_prec_fun<is_row_vec_y, is_row_vec_mu>(1, 1),
                           get_vvar(y_), get_vvar(mu_), get_vvar(sigma_));
  }
}

TEST(MultiNormal, TestGradFunctionalVectorized) {
  test_all<1,1>();
  test_all<1,-1>();
  test_all<-1,1>();
  test_all<-1,-1>();
}
