#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <set>
#include <vector>

// test solver functions
// was part of test/unit/lang/ast_test.hpp

using stan::lang::algebra_solver;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::double_type;
using stan::lang::expression;
using stan::lang::int_type;
using stan::lang::integrate_ode;
using stan::lang::vector_type;
using stan::lang::variable;

TEST(langAst, solveOde) {

  integrate_ode so; // null ctor should work and not raise error

  std::string integration_function_name = "bar";
  std::string system_function_name = "foo";

  double_type tDouble;
  double_type tDoubleData(true);
  int_type tIntData(true);

  variable y0("y0_var_name");
  y0.set_type(bare_array_type(tDouble, 1));

  variable t0("t0_var_name");
  t0.set_type(tDouble);

  variable ts("ts_var_name");
  ts.set_type(bare_array_type(tDoubleData, 1));

  variable theta("theta_var_name");
  theta.set_type(bare_array_type(tDouble, 1));

  variable x("x_var_name");
  x.set_type(bare_array_type(tDoubleData, 1));
  
  variable x_int("x_int_var_name");
  x_int.set_type(bare_array_type(tIntData, 1));

  // example of instantiation
  integrate_ode so2(integration_function_name, system_function_name,
                    y0, t0, ts, theta, x, x_int);

  // dumb test to make sure we at least get the right types back
  EXPECT_EQ(integration_function_name, so2.integration_function_name_);
  EXPECT_EQ(system_function_name, so2.system_function_name_);
  EXPECT_EQ(y0.type_, so2.y0_.bare_type());
  EXPECT_EQ(t0.type_, so2.t0_.bare_type());
  EXPECT_EQ(ts.type_, so2.ts_.bare_type());
  EXPECT_EQ(theta.type_, so2.theta_.bare_type());
  EXPECT_EQ(x.type_, so2.x_.bare_type());
  EXPECT_EQ(x_int.type_, so2.x_int_.bare_type());

  expression e2(so2);
  EXPECT_EQ(bare_expr_type(bare_array_type(tDouble,2)), e2.bare_type());
}

TEST(langAst, solveAlgebra) {
    algebra_solver so;  // null ctor should work and not raise error
    std::string system_function_name = "bronzino";

    double_type tDouble;
    double_type tDoubleData(true);
    int_type tIntData(true);
    vector_type tVector;
    vector_type tVectorData(true);
    
    variable y("y_var_name");
    y.set_type(tVector);  // vector from Eigen
    
    variable theta("theta_var_name");
    theta.set_type(tVector);
    
    variable x_r("x_r_r_var_name");
    x_r.set_type(bare_array_type(tDoubleData, 1));
    
    variable x_i("x_i_var_name");
    x_i.set_type(bare_array_type(tIntData, 1));
    
    // example of instantiation
    algebra_solver so2(system_function_name, y, theta, x_r, x_i);
    
    // dumb test to make sure we at least get the right types back
    EXPECT_EQ(system_function_name, so2.system_function_name_);
    EXPECT_EQ(y.type_, so2.y_.bare_type());
    EXPECT_EQ(theta.type_, so2.theta_.bare_type());
    EXPECT_EQ(x_r.type_, so2.x_r_.bare_type());
    EXPECT_EQ(x_i.type_, so2.x_i_.bare_type());
    
    expression e2(so2);
    EXPECT_EQ(bare_expr_type(tVector), e2.bare_type());
}
