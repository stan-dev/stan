#include <stan/math/prim/mat/fun/stan_print.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/fwd/scal/fun/operator_addition.hpp>
#include <stan/math/fwd/scal/fun/operator_division.hpp>
#include <stan/math/fwd/scal/fun/operator_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_multiplication.hpp>
#include <stan/math/fwd/scal/fun/operator_not_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_subtraction.hpp>
#include <stan/math/fwd/scal/fun/operator_unary_minus.hpp>
#include <stan/math/rev/scal/fun/operator_addition.hpp>
#include <stan/math/rev/scal/fun/operator_divide_equal.hpp>
#include <stan/math/rev/scal/fun/operator_division.hpp>
#include <stan/math/rev/scal/fun/operator_equal.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_less_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_minus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_multiplication.hpp>
#include <stan/math/rev/scal/fun/operator_multiply_equal.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/operator_plus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>
#include <stan/math/rev/scal/fun/operator_unary_decrement.hpp>
#include <stan/math/rev/scal/fun/operator_unary_increment.hpp>
#include <stan/math/rev/scal/fun/operator_unary_negative.hpp>
#include <stan/math/rev/scal/fun/operator_unary_not.hpp>
#include <stan/math/rev/scal/fun/operator_unary_plus.hpp>

TEST(AgradFwdMatrixStanPrint,fvar_double) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::stan_print;
  using stan::agrad::fvar;

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
  using stan::agrad::fvar;

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
  EXPECT_EQ("1",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<var> > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1,1,1]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1,1],[1,1]]",output.str());
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
  EXPECT_EQ("1",output.str());

  output.str( std::string() );
  output.clear();
  std::vector<fvar<fvar<var> > > b;
  b.push_back(a);
  b.push_back(a);
  b.push_back(a);
  stan_print(&output, b);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1> c(3);
  c << a,a,a;
  stan_print(&output, c);
  EXPECT_EQ("[1,1,1]",output.str());
          
  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, 1, Eigen::Dynamic> d(3);
  d << a,a,a;
  stan_print(&output, d);
  EXPECT_EQ("[1,1,1]",output.str());

  output.str( std::string() );
  output.clear();
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> e(2,2);
  e << a,a,a,a;
  stan_print(&output, e);
  EXPECT_EQ("[[1,1],[1,1]]",output.str());
}
