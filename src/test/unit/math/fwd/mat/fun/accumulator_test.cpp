#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/accumulator.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;

// test sum of first n numbers for sum of a
template <typename T>
void test_sum(stan::math::accumulator<T>& a,
              int n) {
  EXPECT_TRUE((n * (n + 1)) / 2 == a.sum());
}

TEST(AgradFwdMatrixAccumulate,fvar_double) {
  using stan::math::accumulator;
  
  accumulator<fvar<double> > a;
  test_sum(a, 0);

  a.add(1.0);
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  test_sum(a, 1000);
  
}
TEST(AgradFwdMatrixAccumulate,collection_fvar_double) {

  using stan::math::accumulator;
  using std::vector;
  using stan::math::vector_fd;
  using stan::math::matrix_fd;

  accumulator<fvar<double> > a;

  int pos = 0;
  test_sum(a, 0);

  vector<fvar<double> > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);                                         
  test_sum(a, pos-1);

  a.add(pos++);                    
  test_sum(a, pos-1);

  double x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  vector<vector<fvar<double> > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<double> > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  matrix_fd m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  vector_fd mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<vector_fd> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_fd vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);
}

TEST(AgradFwdMatrixAccumulate,fvar_fvar_double) {
  using stan::math::accumulator;
  
  accumulator<fvar<fvar<double> > > a;
  test_sum(a, 0);

  a.add(1.0);
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  test_sum(a, 1000);
  
}
TEST(AgradFwdMatrixAccumulate,collection_fvar_fvar_double) {

  using stan::math::accumulator;
  using std::vector;
  using stan::math::vector_ffd;
  using stan::math::matrix_ffd;

  accumulator<fvar<fvar<double> > > a;

  int pos = 0;
  test_sum(a, 0);

  vector<fvar<fvar<double> > > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);                                         
  test_sum(a, pos-1);

  a.add(pos++);                    
  test_sum(a, pos-1);

  double x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  vector<vector<fvar<fvar<double> > > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<fvar<double> > > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  matrix_ffd m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  vector_ffd mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<vector_ffd> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_ffd vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);
}
