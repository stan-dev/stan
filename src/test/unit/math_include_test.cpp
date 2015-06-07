#include <stan/math.hpp>
#include <cmath>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

// This test fixture swallows output to std::cout
class Math : public ::testing::Test {
public:
  void SetUp() {
    output_.str("");
    cout_backup_ = std::cout.rdbuf();
    std::cout.rdbuf(output_.rdbuf());
  }

  void TearDown() {
    std::cout.rdbuf(cout_backup_);
    stan::math::recover_memory();
  }

  std::stringstream output_;
  std::streambuf *cout_backup_;
};

TEST_F(Math, paper_example_1) {
  using std::pow;
  double y = 1.3;
  stan::math::var mu = 0.5, sigma = 1.2;
    
  stan::math::var lp = 0;
  lp -= pow(2 * stan::math::pi(), -0.5);
  lp -= log(sigma);
  lp -= 0.5 * pow((y - mu) / sigma, 2);
  std::cout << "f(mu,sigma)= " << lp.val() << std::endl;

  lp.grad();
  std::cout << " d.f / d.mu = " << mu.adj()
            << " d.f / d.sigma = " << sigma.adj() << std::endl;
}

TEST_F(Math, paper_example_2) {
  double y = 1.3;
  stan::math::var mu = 0.5, sigma = 1.2;
  
  stan::math::var lp = 0;
  lp -= pow(2 * boost::math::constants::pi<double>(), -0.5);
  lp -= log(sigma);
  lp -= 0.5 * pow((y - mu) / sigma, 2);

  std::vector<stan::math::var> theta;
  theta.push_back(mu);   theta.push_back(sigma);
  std::vector<double> g;
  lp.grad(theta, g);
  std::cout << " d.f / d.mu = " << g[0]
            << " d.f / d.sigma = " << g[1] << std::endl;
}

namespace paper {  // paper_example_3
template <typename T1, typename T2, typename T3>
inline typename boost::math::tools::promote_args<T1, T2, T3>::type
normal_log(const T1& y, const T2& mu, const T3& sigma) {
  using std::pow;  using std::log;  
  return -0.5 * pow((y - mu) / sigma, 2.0)
    - log(sigma)
    - 0.5 * log(2 * stan::math::pi());
}
}

TEST_F(Math, paper_example_3) {
  double y = 1.3;
  stan::math::var mu = 0.5, sigma = 1.2;

  stan::math::var lp = normal_log(y, mu, sigma);
  EXPECT_FLOAT_EQ(-1.323482, lp.val());
}

namespace paper {  // paper_example_4: remove 'paper::' when including in the paper
using Eigen::Matrix;  
using Eigen::Dynamic;

struct normal_ll {
  const Matrix<double, Dynamic, 1> y_;

  normal_ll(const Matrix<double, Dynamic, 1>& y) : y_(y) { }

  template <typename T>
  T operator()(const Matrix<T, Dynamic, 1>& theta) const {
    T mu = theta[0];   
    T sigma = theta[1];
    T lp = 0;
    for (int n = 0; n < y_.size(); ++n)
      lp += paper::normal_log(y_[n], mu, sigma);
    return lp;
  }
};
}

TEST_F(Math, paper_example_4) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  using paper::normal_ll;

  Matrix<double, Dynamic, 1> y(3);
  y << 1.3, 2.7, -1.9;
  normal_ll f(y);
  
  Matrix<double, Dynamic, 1> theta(2);
  theta << 1.3, 2.9;
  
  double fx;
  Matrix<double, Dynamic, 1> grad_fx;
  stan::math::gradient(f, theta, fx, grad_fx);
}

namespace paper_example_5 {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  
  struct functor {
    const Matrix<double, Dynamic, 1> y_;
    
    functor(const Matrix<double, Dynamic, 1>& y) : y_(y) { }
    
    template <typename T>
    Matrix<T, Dynamic, 1> operator()(const Matrix<T, Dynamic, 1>& theta) const {
      Matrix<T, Dynamic, 1> lp(y_.size());
      T mu = theta[0];   
      T sigma = theta[1];
      for (int n = 0; n < y_.size(); ++n)
        lp[n] = paper::normal_log(y_[n], mu, sigma);
      return lp;
    }
  };
  
}

TEST_F(Math, paper_example_5) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  Matrix<double, Dynamic, 1> y(3);
  y << 1.3, 2.7, -1.9;
  paper_example_5::functor f(y);
  
  Matrix<double, Dynamic, 1> x(2);
  x << 1.3, 2.9;

  
  // paper_example_5 starts with the next line
  // Matrix<double, Dynamic, 1> x = ...;   // inputs
  
  Matrix<var, Dynamic, 1> x_var(x.size());
  for (int i = 0; i < x.size(); ++i) x_var(i) = x(i);
  
  Matrix<var, Dynamic, 1> f_x_var = f(x_var);
  
  Matrix<double, Dynamic, 1> f_x(f_x_var.size());
  for (int i = 0; i < f_x.size(); ++i) f_x(i) = f_x_var(i).val();
  
  Matrix<double, Dynamic, Dynamic> J(f_x_var.size(), x_var.size());
  for (int i = 0; i < f_x_var.size(); ++i) {
    if (i > 0) stan::math::set_zero_all_adjoints();
    f_x_var(i).grad();
    for (int j = 0; j < x_var.size(); ++j)
      J(i,j) = x_var(j).adj();
  }
}

TEST_F(Math, paper_example_6) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  Matrix<double, Dynamic, 1> y(3);
  y << 1.3, 2.7, -1.9;
  paper_example_5::functor f(y);
  
  Matrix<double, Dynamic, 1> x(2);
  x << 1.3, 2.9;

  // paper_example_6
  Matrix<double, Dynamic, Dynamic> J;
  Matrix<double, Dynamic, 1> f_x;
  stan::math::jacobian(f, x, f_x, J);
}
