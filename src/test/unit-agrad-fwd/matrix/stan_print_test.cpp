#include <stan/math/matrix/stan_print.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>  

TEST(AgradFwdMatrixStanPrint,fvar_double) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::agrad::fvar;

  std::stringstream output;
  fvar<double> a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1:2",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<double> > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1:2,1:2,1:2]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1:2,1:2,1:2]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1:2,1:2,1:2]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<double>, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1:2,1:2],[1:2,1:2]]",output.str());
}

TEST(AgradFwdMatrixStanPrint,fvar_fvar_double) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::agrad::fvar;

  std::stringstream output;
  fvar<fvar<double> > a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1:0:2:0",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<fvar<double> > > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1:0:2:0,1:0:2:0],[1:0:2:0,1:0:2:0]]",output.str());
}

TEST(AgradFwdMatrixStanPrint,fvar_var) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::stringstream output;
  fvar<var> a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1:0:2:0",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<var> > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1:0:2:0,1:0:2:0,1:0:2:0]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1:0:2:0,1:0:2:0],[1:0:2:0,1:0:2:0]]",output.str());
}

TEST(AgradFwdMatrixStanPrint,fvar_fvar_var) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::stringstream output;
  fvar<fvar<var> > a(1,2);
  stan_print(&output, a);
  EXPECT_EQ("1:0:0:0:2:0:0:0",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<fvar<var> > > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0],[1:0:0:0:2:0:0:0,1:0:0:0:2:0:0:0]]",output.str());
}
