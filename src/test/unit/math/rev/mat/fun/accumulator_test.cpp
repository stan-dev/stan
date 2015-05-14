#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/accumulator.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

// test sum of first n numbers for sum of a
void test_sum(stan::math::accumulator<stan::math::var>& a,
              int n) {
  EXPECT_FLOAT_EQ((n * (n + 1)) / 2, a.sum().val());
}

TEST(AgradRevMatrix,accumulateDouble) {
  using stan::math::accumulator;
  using stan::math::var;

  accumulator<var> a;
  test_sum(a, 0);

  a.add(var(1.0));
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(var(i));
  test_sum(a, 1000);
  
}
TEST(AgradRevMathMatrix,accumulateCollection) {
  // tests int, double, vector<double>, vector<int>,
  // Matrix<double,...>,
  // var, vector<var>, Matrix<var,...>, 
  // and recursions of vector<T>

  using stan::math::accumulator;
  using std::vector;
  using Eigen::VectorXd;
  using Eigen::MatrixXd;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  accumulator<var> a;

  int pos = 0;
  test_sum(a, 0);

  vector<var> v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = var(pos++);
  a.add(v);                                         
  test_sum(a, pos-1);

  vector<double> d(10);
  for (size_t i = 0; i < 10; ++i)
    d[i] = pos++;
  a.add(d);                                         
  test_sum(a, pos-1);

  var x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  int nnn = pos++;
  a.add(nnn);
  test_sum(a, pos-1);

  double xxx = pos++;
  a.add(xxx);
  test_sum(a, pos-1);

  vector<int> u(10);         
  for (size_t i = 0; i < 10; ++i)
    a.add(pos++);
  test_sum(a, pos-1);

  vector<vector<int> > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<int> w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  MatrixXd m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  VectorXd mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<VectorXd> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    VectorXd vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);

  Matrix<var,Dynamic,Dynamic> mvar(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      mvar(i,j) = pos++;
  a.add(mvar);
  test_sum(a, pos-1);

  Matrix<var,1,Dynamic> mvvar(7);
  for (int i = 0; i < 7; ++i)
    mvvar(i) = pos++;
  a.add(mvvar);
  test_sum(a, pos-1);

  vector<Matrix<var,Dynamic,1> > vvx_var(8);
  for (size_t i = 0; i < 8; ++i) {
    Matrix<var,Dynamic,1> vx_var(3);
    for (int j = 0; j < 3; ++j)
      vx_var(j) = pos++;
    vvx_var[i] = vx_var;
  }
  a.add(vvx_var);
  test_sum(a, pos-1);
}



