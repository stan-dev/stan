#include <iostream>
#include <sstream>
#include <boost/random/additive_combine.hpp>
#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <stan/io/dump.hpp>
#include <test/test-models/good/lang/test_lp.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>

void test_generate_quoted_string(const std::string& s,
                                 const std::string& quoted_s) {
  std::stringstream ss;
  stan::lang::generate_quoted_string(s, ss);
  EXPECT_EQ(quoted_s, ss.str());
}

void test_generate_quoted_string_quote(const std::string& s,
                           const std::string& expected_output_content) {
  std::stringstream ss;
  stan::lang::generate_quoted_string(s,ss);
  std::string s_rendered = ss.str();
  EXPECT_EQ("\"" + expected_output_content + "\"", ss.str());
}

TEST(langGenerator, quotedString) {
  test_generate_quoted_string_quote("","");
  test_generate_quoted_string_quote("abc", "abc");
  test_generate_quoted_string_quote("abc'def", "abc\\'def");
  test_generate_quoted_string_quote("\"abc", "\\\"abc");
  test_generate_quoted_string_quote("abc\"", "abc\\\"");
  test_generate_quoted_string_quote("abc\"def", "abc\\\"def");
  test_generate_quoted_string_quote("abc\"def\"ghi", "abc\\\"def\\\"ghi");
}

void test_generate_quoted_expression(const stan::lang::expression& e,
                                     const std::string& e_exp) {
  std::stringstream ss;
  stan::lang::generate_quoted_expression(e, ss);
  EXPECT_EQ(e_exp, ss.str());
}

TEST(lang,printStringLiteral) {
  test_generate_quoted_string("", "\"\"");
  test_generate_quoted_string("\\d\\", "\"\\\\d\\\\\"");
  test_generate_quoted_string("ab\"c", "\"ab\\\"c\"");
  test_generate_quoted_string("'hey,' he said.","\"\\'hey,\\' he said.\"");
}

TEST(lang,printQuotedExpression) {
  using stan::lang::expression;
  using stan::lang::index_op;
  using stan::lang::int_literal;
  using stan::lang::variable;
  using std::vector;
  test_generate_quoted_expression(int_literal(1), "\"1\"");
  vector<expression> args;

  expression expr(variable("foo"));
  vector<vector<expression> > dimss;
  vector<expression> dim;
  dim.push_back(int_literal(1));
  dimss.push_back(dim);
  std::string s_exp = "\"get_base1(foo,1,\\\"foo\\\",1)\"";
  test_generate_quoted_expression(index_op(expr,dimss), s_exp);
}


TEST(lang, logProbPolymorphismDouble) {
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

  // only test generate_inits for doubles -- no var allowed
  std::string init_txt = "y <- c(-2.9,1.2)";
  std::stringstream init_in(init_txt);
  stan::io::dump init_dump(init_in);
  std::vector<int> params_i_init;
  std::vector<double> params_r_init;
  std::stringstream pstream;
  model.transform_inits(init_dump, params_i_init, params_r_init, &pstream);
  EXPECT_EQ(0U, params_i_init.size());
  EXPECT_EQ(2U, params_r_init.size());

  Matrix<double,Dynamic,1> params_r_vec_init;
  model.transform_inits(init_dump, params_r_vec_init, &pstream);
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

  boost::ecuyer1988 rng(123);
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
TEST(lang, logProbPolymorphismVar) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

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

TEST(lang, generate_model_typedef) {
  std::string model_name = "name";
  std::stringstream ss;
  stan::lang::generate_model_typedef(model_name,ss);

  EXPECT_EQ(1, count_matches("typedef name_namespace::name stan_model;",
                             ss.str()));
}

// * transform_inits

TEST(lang, generate_cpp) {
  stan::lang::program prog;
  std::string model_name = "m";
  std::stringstream output;

  stan::lang::generate_cpp(prog, model_name, output);
  std::string output_str = output.str();

  EXPECT_EQ(1, count_matches("// Code generated by Stan version ", output_str))
    << "generate_version_comment()";
  EXPECT_EQ(1, count_matches("#include", output_str))
    << "generate_includes()";
  EXPECT_EQ(1, count_matches("namespace " + model_name + "_namespace {", output_str))
    << "generate_start_namespace()";
  EXPECT_LT(1, count_matches("using", output_str))
    << "generate_usings()";
  EXPECT_EQ(3, count_matches("typedef Eigen::Matrix", output_str))
    << "generate_typedefs()";

  // << "generate_functions()";

  EXPECT_EQ(1, count_matches("class " + model_name, output_str))
    << "generate_class_decl()";
  EXPECT_EQ(1, count_matches("private:", output_str))
     << "generate_private_decl()";

  // << "generate_member_var_decls()";
  // << "generate_member_var_decls()";

  EXPECT_EQ(1, count_matches("public:", output_str))
    << "generate_public_decl()";
  // FIXME(carpenter): change this again when the second ctor eliminated
  EXPECT_EQ(2, count_matches(" " + model_name + "(", output_str))
    << "generate_constructor()";
  EXPECT_EQ(1, count_matches("~" + model_name + "(", output_str))
    << "generate_destructor()";
  EXPECT_EQ(2, count_matches("void transform_inits(", output_str))
    << "generate_init_method()";
  EXPECT_EQ(1, count_matches("T__ log_prob(", output_str))
    << "generate_log_prob()";
  EXPECT_EQ(1, count_matches("T_ log_prob(", output_str))
    << "generate_log_prob()";
  EXPECT_EQ(1, count_matches("void get_param_names(", output_str))
    << "generate_param_names_method()";
  EXPECT_EQ(1, count_matches("void get_dims(", output_str))
    << "generate_dims_method()";
  EXPECT_EQ(2, count_matches("void write_array(", output_str))
    << "generate_write_array_method()";
  EXPECT_EQ(1, count_matches("static std::string model_name()", output_str))
    << "generate_model_name_method()";
  EXPECT_EQ(1, count_matches("void constrained_param_names(", output_str))
    << "generate_constrained_param_names_method()";
  EXPECT_EQ(1, count_matches("void unconstrained_param_names(", output_str))
    << "generate_unconstrained_param_names_method()";
  EXPECT_EQ(1, count_matches("}; // model", output_str))
    << "generate_end_class_decl()";
  EXPECT_EQ(1, count_matches("typedef " + model_name + "_namespace::"
                             + model_name + " stan_model;",
                             output_str))
    << "generate_model_typedef()";


  EXPECT_EQ(0, count_matches("int main", output_str));
}

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


TEST(langGenerator, sliceIndexes) {
  // boundary condition of no indices
  std::vector<stan::lang::idx> is2;
  std::stringstream o2;
  stan::lang::generate_idxs(is2, o2);
  EXPECT_EQ("stan::model::nil_index_list()", o2.str());

  // two indexes
  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui3(e_int3);
  stan::lang::idx idx0(ui3);

  stan::lang::expression e_int5(stan::lang::int_literal(5));
  stan::lang::ub_idx ub5(e_int5);
  stan::lang::idx idx1(ub5);

  std::vector<stan::lang::idx> is;
  is.push_back(idx0);
  is.push_back(idx1);

  std::stringstream o;
  stan::lang::generate_idxs(is, o);
  EXPECT_EQ("stan::model::cons_list(stan::model::index_uni(3), stan::model::cons_list(stan::model::index_max(5), stan::model::nil_index_list()))",
            o.str());
}

TEST(langGenerator, slicedAssigns) {
  using stan::lang::DOUBLE_T;

  stan::lang::variable v("foo");
  v.set_type(DOUBLE_T, 0);

  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui3(e_int3);
  stan::lang::idx idx0(ui3);

  stan::lang::expression e_int5(stan::lang::int_literal(5));
  stan::lang::ub_idx ub5(e_int5);
  stan::lang::idx idx1(ub5);

  std::vector<stan::lang::idx> is;
  is.push_back(idx0);
  is.push_back(idx1);

  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::assgn a(v, is, e);
  stan::lang::statement s(a);
  s.begin_line_ = 12U;
  s.end_line_ = 14U;

  std::stringstream o;
  generate_statement(s, 2, o, true, true, false);
  EXPECT_TRUE(0U < o.str().find(
      "stan::model::cons_list(stan::model::index_uni(3), stan::model::cons_list(stan::model::index_max(5), stan::model::nil_index_list()))"));
}
TEST(langGenerator, fills) {
  expect_matches(1,
                 "transformed data { int a[3]; }"
                 " model { }",
                 "stan::math::fill(a, std::numeric_limits<int>::min());\n");
}


TEST(langGenerator, genRealVars) {
  using stan::lang::scope;
  using stan::lang::transformed_data_origin;
  using stan::lang::function_argument_origin;
  scope td_origin = transformed_data_origin;
  scope fun_origin = function_argument_origin;
  std::stringstream o;

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, true, o);
  EXPECT_EQ(1, count_matches("T__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, true, o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, false, o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, false, o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, true, o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, true, o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, false, o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, false, o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));
}

TEST(langGenerator, genArrayVars) {
  using stan::lang::base_expr_type;
  using stan::lang::INT_T;
  using stan::lang::DOUBLE_T;
  using stan::lang::VECTOR_T;
  using stan::lang::ROW_VECTOR_T;
  using stan::lang::MATRIX_T;
  using stan::lang::scope;
  using stan::lang::transformed_data_origin;
  using stan::lang::function_argument_origin;
  scope td_origin = transformed_data_origin;
  scope fun_origin = function_argument_origin;
  std::stringstream ssReal;
  std::stringstream o;

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, true, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("T__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, true, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, false, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, false, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, true, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, true, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, false, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, false, ssReal);
  stan::lang::generate_array_var_type(DOUBLE_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("fun_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(INT_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("int", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(INT_T,ssReal.str(),false,o);
  EXPECT_EQ(1, count_matches("int", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(VECTOR_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<T__,Eigen::Dynamic,1> ", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(VECTOR_T,ssReal.str(),false,o);
  EXPECT_EQ(1, count_matches("vector_d", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(ROW_VECTOR_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<T__,1,Eigen::Dynamic> ", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(ROW_VECTOR_T,ssReal.str(),false,o);
  EXPECT_EQ(1, count_matches("row_vector_d", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(MATRIX_T,ssReal.str(),true,o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> ", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_array_var_type(MATRIX_T,ssReal.str(),false,o);
  EXPECT_EQ(1, count_matches("matrix_d", o.str()));
}

TEST(genArrayBuilderAdds, addScalars) {
  stan::lang::expression e_d3(stan::lang::double_literal(3));
  std::vector<stan::lang::expression> elts;
  elts.push_back(e_d3);
  elts.push_back(e_d3);
  elts.push_back(e_d3);
  std::stringstream o2;
  stan::lang::generate_array_builder_adds(elts, true, false, o2);
  EXPECT_EQ(3, count_matches(".add(", o2.str()));
}
