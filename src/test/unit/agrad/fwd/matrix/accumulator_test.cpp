#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/accumulator.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

using stan::agrad::fvar;
using stan::agrad::var;

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
  using stan::agrad::vector_fd;
  using stan::agrad::matrix_fd;

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
  using stan::agrad::vector_ffd;
  using stan::agrad::matrix_ffd;

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


TEST(AgradFwdMatrixAccumulate,fvar_var) {
  using stan::math::accumulator;
  
  accumulator<fvar<var> > a;
  test_sum(a, 0);

  a.add(1.0);
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  test_sum(a, 1000);
  
}
TEST(AgradFwdMatrixAccumulate,collection_fvar_var) {

  using stan::math::accumulator;
  using std::vector;
  using stan::agrad::vector_fv;
  using stan::agrad::matrix_fv;

  accumulator<fvar<var> > a;

  int pos = 0;
  test_sum(a, 0);

  vector<fvar<var> > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);                                         
  test_sum(a, pos-1);

  a.add(pos++);                    
  test_sum(a, pos-1);

  double x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  vector<vector<fvar<var> > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<var> > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  matrix_fv m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  vector_fv mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<vector_fv> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_fv vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);
}

TEST(AgradFwdMatrixAccumulate,fvar_fvar_var) {
  using stan::math::accumulator;
  
  accumulator<fvar<fvar<var> > > a;
  test_sum(a, 0);

  a.add(1.0);
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  test_sum(a, 1000);
  
}
TEST(AgradFwdMatrixAccumulate,collection_fvar_fvar_var) {

  using stan::math::accumulator;
  using std::vector;
  using stan::agrad::vector_ffv;
  using stan::agrad::matrix_ffv;

  accumulator<fvar<fvar<var> > > a;

  int pos = 0;
  test_sum(a, 0);

  vector<fvar<fvar<var> > > v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);                                         
  test_sum(a, pos-1);

  a.add(pos++);                    
  test_sum(a, pos-1);

  int x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  vector<vector<fvar<fvar<var> > > > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<fvar<fvar<var> > > w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  matrix_ffv m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  vector_ffv mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<vector_ffv> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    vector_ffv vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);
}
