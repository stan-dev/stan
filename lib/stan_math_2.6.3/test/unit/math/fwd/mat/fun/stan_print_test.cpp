#include <stan/math/prim/mat/fun/stan_print.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixStanPrint,fvar_double) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::math::fvar;

  std::stringstream output;
  fvar<double> a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<double> > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1,1,1]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1,1],[1,1]]",output.str());
}

TEST(AgradFwdMatrixStanPrint,fvar_fvar_double) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::math::fvar;

  std::stringstream output;
  fvar<fvar<double> > a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<fvar<double> > > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1,1,1]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1,1],[1,1]]",output.str());
}
