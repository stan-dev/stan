#include <iostream>
#include <sstream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <stan/agrad/rev.hpp>
#include <stan/gm/ast.hpp>
#include <stan/gm/generator.hpp>
#include <stan/io/dump.hpp>
#include <test/test-models/good/gm/test_lp.cpp>
#include <test/unit/gm/utility.hpp>
#include <gtest/gtest.h>

void test_print_string_literal(const std::string& s,
                               const std::string& s_exp) {
  std::stringstream ss;
  stan::gm::print_string_literal(ss,s);
  EXPECT_EQ(s_exp, ss.str());
}

void test_print_quoted_expression(const stan::gm::expression& e,
                                  const std::string& e_exp) {
  std::stringstream ss;
  stan::gm::print_quoted_expression(ss,e);
  EXPECT_EQ(e_exp, ss.str());
}

TEST(gm,printStringLiteral) {
  test_print_string_literal("","\"\"");
  test_print_string_literal("\\d\\","\"\\\\d\\\\\"");
  test_print_string_literal("ab\"c", "\"ab\\\"c\"");
  test_print_string_literal("'hey,' he said.","\"\\'hey,\\' he said.\"");
}

TEST(gm,printQuotedExpression) {
  using stan::gm::expression;
  using stan::gm::index_op;
  using stan::gm::int_literal;
  using stan::gm::variable;
  using std::vector;
  test_print_quoted_expression(int_literal(1),"\"1\"");
  vector<expression> args;

  expression expr(variable("foo"));
  vector<vector<expression> > dimss;
  vector<expression> dim;
  dim.push_back(int_literal(1));
  dimss.push_back(dim);
  std::string s_exp = "\"get_base1(foo,1,\\\"foo\\\",1)\"";
  test_print_quoted_expression(index_op(expr,dimss),
                               s_exp);
}


TEST(gm, logProbPolymorphismDouble) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_model_namespace::test_lp_model model(dump);

  std::vector<double> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;
  
  Matrix<double, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  double lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  double lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  // only test write_csv for doubles -- no var allowed
  std::stringstream s1;
  std::stringstream s2;
  boost::ecuyer1988 rng(123);
  model.write_csv(rng,params_r,params_i,s1,0);
  model.write_csv(rng,params_r_vec,s2,0);
  EXPECT_EQ(s1.str(), s2.str());

  // only test generate_inits for doubles -- no var allowed
  std::string init_txt = "y <- c(-2.9,1.2)";
  std::stringstream init_in(init_txt);
  stan::io::dump init_dump(init_in);
  std::vector<int> params_i_init;
  std::vector<double> params_r_init;
  model.transform_inits(init_dump, params_i_init, params_r_init);
  EXPECT_EQ(0U, params_i_init.size());
  EXPECT_EQ(2U, params_r_init.size());

  Matrix<double,Dynamic,1> params_r_vec_init;
  model.transform_inits(init_dump, params_r_vec_init);
  EXPECT_EQ(int(params_r.size()), params_r_vec_init.size());
  for (int i = 0; i < params_r_vec_init.size(); ++i)
    EXPECT_FLOAT_EQ(params_r_init[i], params_r_vec_init(i));

  // only test write_array for doubles --- no var allowed
  std::vector<double> params_r_write(2);
  params_r_write[0] = -3.2;  
  params_r_write[1] = 1.79;
  std::vector<int> params_i_write;
  
  Matrix<double,Dynamic,1> params_r_vec_write(2);
  params_r_vec_write << -3.2, 1.79;

  for (int incl_tp = 0; incl_tp < 2; ++incl_tp) {
    for (int incl_gq = 0; incl_gq < 2; ++incl_gq) { 
      std::vector<double> vars_write;
      Matrix<double,Dynamic,1> vars_vec_write(17);
      model.write_array(rng, params_r_write, params_i_write, vars_write, incl_tp, incl_gq, 0);
      model.write_array(rng, params_r_vec_write, vars_vec_write, incl_tp, incl_gq, 0);
      EXPECT_EQ(int(vars_write.size()), vars_vec_write.size());
      for (int i = 0; i < vars_vec_write.size(); ++i)
        EXPECT_FLOAT_EQ(vars_write[i], vars_vec_write(i));
    }
  }

}
TEST(gm, logProbPolymorphismVar) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_model_namespace::test_lp_model model(dump);

  std::vector<var> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;
  
  Matrix<var, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  var lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  var lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());
}

TEST(gm, generate_model_typedef) {
  std::string model_name = "name";
  std::stringstream ss;
  stan::gm::generate_model_typedef(model_name,ss);
  
  EXPECT_EQ(1, count_matches("typedef name_namespace::name stan_model;", 
                             ss.str()));
}

// * write_csv
// * transform_inits

TEST(gm, generate_cpp) {
  stan::gm::program prog;
  std::string model_name = "m";
  std::stringstream output;

  stan::gm::generate_cpp(prog, model_name, output);
  
  EXPECT_EQ(1, count_matches("// Code generated by Stan version ", output.str()))
    << "generate_version_comment()";
  EXPECT_EQ(2, count_matches("#include", output.str()))
    << "generate_includes()";
  EXPECT_EQ(1, count_matches("namespace " + model_name + "_namespace {", output.str()))
    << "generate_start_namespace()";
  EXPECT_LT(1, count_matches("using", output.str()))
    << "generate_usings()";
  EXPECT_EQ(3, count_matches("typedef Eigen::Matrix", output.str()))
    << "generate_typedefs()";

  // << "generate_functions()";

  EXPECT_EQ(1, count_matches("class " + model_name, output.str()))
    << "generate_class_decl()";
  EXPECT_EQ(1, count_matches("private:", output.str()))
     << "generate_private_decl()";

  // << "generate_member_var_decls()";
  // << "generate_member_var_decls()";

  EXPECT_EQ(1, count_matches("public:", output.str()))
    << "generate_public_decl()";
  EXPECT_EQ(1, count_matches(" " + model_name + "(", output.str()))  
    << "generate_constructor()";
  EXPECT_EQ(1, count_matches("~" + model_name + "(", output.str()))  
    << "generate_destructor()";
  EXPECT_EQ(2, count_matches("void transform_inits(", output.str()))
    << "generate_init_method()";
  EXPECT_EQ(1, count_matches("T__ log_prob(", output.str()))
    << "generate_log_prob()";
  EXPECT_EQ(1, count_matches("T_ log_prob(", output.str()))
    << "generate_log_prob()";
  EXPECT_EQ(1, count_matches("void get_param_names(", output.str()))
    << "generate_param_names_method()";
  EXPECT_EQ(1, count_matches("void get_dims(", output.str()))
    << "generate_dims_method()";
  EXPECT_EQ(2, count_matches("void write_array(", output.str()))
    << "generate_write_array_method()";
  EXPECT_EQ(1, count_matches("void write_csv_header(", output.str()))
    << "generate_write_csv_header_method()";
  EXPECT_EQ(2, count_matches("void write_csv(", output.str()))
    << "generate_write_csv_method()";
  EXPECT_EQ(1, count_matches("static std::string model_name()", output.str()))
    << "generate_model_name_method()";
  EXPECT_EQ(1, count_matches("void constrained_param_names(", output.str()))
    << "generate_constrained_param_names_method()";
  EXPECT_EQ(1, count_matches("void unconstrained_param_names(", output.str()))
    << "generate_unconstrained_param_names_method()";
  EXPECT_EQ(1, count_matches("}; // model", output.str()))
    << "generate_end_class_decl()";
  EXPECT_EQ(1, count_matches("} // namespace", output.str()))
    << "generate_end_namespace()";
  EXPECT_EQ(1, count_matches("typedef " + model_name + "_namespace::" 
                             + model_name + " stan_model;", 
                             output.str()))
    << "generate_model_typedef()";

  
  EXPECT_EQ(0, count_matches("int main", output.str()));
}

// These next tests depend on the parser to build up a prog instance,
// which is too onerous to do directly from the ast. 
// Very brittle because getting exact match of expected output.

TEST(gmGenerator,funArgsInt0) {
  expect_matches(1,
                  "functions { int foo() { return 3; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(gmGenerator,funArgsInt1Real) {
  expect_matches(1,
                  "functions { int foo(real x) { return 3; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(gmGenerator,funArgsInt1Int) {
  expect_matches(1,
                  "functions { int foo(int x) { return x; } } model { }",
                  "int\n"
                 "foo(");
}
TEST(gmGenerator,funArgs0) {
  expect_matches(1,
                  "functions { real foo() { return 1.7; } } model { }",
                  "double\n"
                 "foo(");
}
TEST(gmGenerator,funArgs1) {
  expect_matches(1,
                 "functions { real foo(real x) { return x; } } model { }",
                 "typename boost::math::tools::promote_args<T0__>::type\n"
                 "foo(");
}
TEST(gmGenerator,funArgs4) {
  expect_matches(1,
                 "functions { real foo(real x1, real x2, real x3, real x4) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__>::type\n"
                 "foo(");
}
TEST(gmGenerator,funArgs5) {
  expect_matches(1,
                 "functions { real foo(real x1, real x2, real x3, real x4, real x5) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, typename boost::math::tools::promote_args<T4__>::type>::type\n"
                 "foo(");
}
TEST(gmGenerator,funArgs0lp) {
  expect_matches(1,
                 "functions { real foo_lp() { return 1.0; } } model { }",
                 "typename boost::math::tools::promote_args<T_lp__>::type\n"
                 "foo_lp(");
}
TEST(gmGenerator,funArgs4lp) {
  expect_matches(1,
                 "functions { real foo_lp(real x1, real x2, real x3, real x4) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, T_lp__>::type\n"
                 "foo_lp(");
}
TEST(gmGenerator,funArgs5lp) {
  expect_matches(1,
                 "functions { real foo_lp(real x1, real x2, real x3, real x4, real x5) { return x1; } } model { }",
                 "typename boost::math::tools::promote_args<T0__, T1__, T2__, T3__, typename boost::math::tools::promote_args<T4__, T_lp__>::type>::type\n"
                 "foo_lp(");
}

TEST(gmGenerator,shortCircuit1) {
  expect_matches(1,
                 "transformed data { int a; a <- 1 || 2; }"
                 "model { }",
                 "(primitive_value(1) || primitive_value(2))");
  expect_matches(1,
                 "transformed data { int a; a <- 1 && 2; }"
                 "model { }",
                 "(primitive_value(1) && primitive_value(2))");


}

