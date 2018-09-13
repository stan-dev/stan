#include <iostream>
#include <sstream>
#include <boost/random/additive_combine.hpp>
#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <stan/io/dump.hpp>
#include <test/test-models/good/lang/test_lp.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>

// These next tests depend on the parser to build up a prog instance,
// which is too onerous to do directly from the ast.
// Very brittle because getting exact match of expected output.

TEST(langGenerator,funArgsInt0) {
  expect_matches(1,
                  "functions { int foo() { return 3; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(langGenerator,funArgsInt1Real) {
  expect_matches(1,
                  "functions { int foo(real x) { return 3; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(langGenerator,funArgsInt1Int) {
  expect_matches(1,
                  "functions { int foo(int x) { return x; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(langGenerator,funArgs0) {
  expect_matches(1,
                  "functions { real foo() { return 1.7; } } model { }",
                  "double\n"
                 "foo(");
}
TEST(langGenerator,funArgs1) {
  expect_matches(1,
                 "functions { real foo(real x) { return x; } } model { }",
                 "typename boost::math::tools::promote_args<T0__>::type\n"
                 "foo(");
}
TEST(langGenerator,funArgs4) {
  expect_matches(1,
                 "functions { real foo(real x1, real x2, real x3, real x4) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__>::type\n"
                 "foo(");
}
TEST(langGenerator,funArgs5) {
  expect_matches(1,
                 "functions { real foo(real x1, real x2, real x3, real x4, real x5) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, typename boost::math::tools::promote_args<T4__>::type>::type\n"
                 "foo(");
}
TEST(langGenerator,funArgs0lp) {
  expect_matches(1,
                 "functions { real foo_lp() { return 1.0; } } model { }",
                 "typename boost::math::tools::promote_args<T_lp__>::type\n"
                 "foo_lp(");
}
TEST(langGenerator,funArgs4lp) {
  expect_matches(1,
                 "functions { real foo_lp(real x1, real x2, real x3, real x4) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, T_lp__>::type\n"
                 "foo_lp(");
}
TEST(langGenerator,funArgs5lp) {
  expect_matches(1,
                 "functions { real foo_lp(real x1, real x2, real x3, real x4, real x5) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, typename boost::math::tools::promote_args<T4__, T_lp__>::type>::type\n"
                 "foo_lp(");
}

TEST(langGenerator,shortCircuit1) {
  expect_matches(1,
                 "transformed data { int a; a <- 1 || 2; }"
                 "model { }",
                 "(primitive_value(1) || primitive_value(2))");
  expect_matches(1,
                 "transformed data { int a; a <- 1 && 2; }"
                 "model { }",
                 "(primitive_value(1) && primitive_value(2))");
}

TEST(langGenerator, fills) {
  expect_matches(1,
                 "transformed data { int a[3]; }"
                 " model { }",
                 "stan::math::fill(a, std::numeric_limits<int>::min());\n");
}

TEST(genExpression, mapRect) {
  std::string model_code
      = "functions {\n"
      "  vector foo(vector shared_params, vector job_params,\n"
      "             real[] data_r, int[] data_i) {\n"
      "    return [1, 2, 3]';\n"
      "  }\n"
      "}\n"
      "data {\n"
      "  vector[3] shared_params_d;\n"
      "  vector[3] job_params_d[3];\n"
      "  real data_r[3, 3];\n"
      "  int data_i[3, 3];\n"
      "}\n"
      "generated quantities {\n"
      "  vector[3] y_hat_gq\n"
      "      = map_rect(foo, shared_params_d, job_params_d, data_r, data_i);\n"
      "}";
  expect_matches(1, model_code, "map_rect<");
  // can't predict number in between
  expect_matches(1, model_code,
                 ", foo_functor__>(shared_params_d,"
                 " job_params_d, data_r, data_i, pstream__)");
}
