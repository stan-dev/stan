#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
#include <boost/variant/polymorphic_get.hpp>
#include <cmath>

#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::idx;
using stan::lang::uni_idx;
using stan::lang::omni_idx;
using stan::lang::expression;
using stan::lang::int_literal;
using stan::lang::function_signatures;
using stan::lang::function_arg_type;
using stan::lang::expr_type;
using stan::lang::base_expr_type;
using stan::lang::ill_formed_type;
using stan::lang::void_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::vector_type;
using stan::lang::row_vector_type;
using stan::lang::matrix_type;
using std::vector;


TEST(langAst, getDefinition) {
  // tests for Stan lang function definitions with fun argument qualifier "data"
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();
  std::string name = "f3args";
  expr_type return_type = expr_type(double_type());
  std::vector<function_arg_type> arg_types;
  arg_types.push_back(function_arg_type(expr_type(double_type(), 2U), true));
  arg_types.push_back(function_arg_type(expr_type(int_type(), 1U)));
  arg_types.push_back(function_arg_type(expr_type(vector_type(), 0U)));

  // check is defined
  fs.add(name, return_type, arg_types);
  stan::lang::function_signature_t sig(return_type, arg_types);
  EXPECT_TRUE(fs.is_defined(name, sig));
}

TEST(langAst, missingDefinition) {
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();

  std::string name = "fmissing";
  expr_type return_type = expr_type(double_type());
  std::vector<function_arg_type> arg_types;
  arg_types.push_back(function_arg_type(expr_type(double_type(), 2U), true));

  // check not defined
  stan::lang::function_signature_t sig(return_type, arg_types);
  EXPECT_FALSE(fs.is_defined(name, sig));
}

TEST(langAst, checkDefinition) {
  // tests for Stan lang function definitions with fun argument qualifier "data"
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();
  std::string name = "f3args";
  expr_type return_type = expr_type(double_type());
  std::vector<function_arg_type> arg_types;
  arg_types.push_back(function_arg_type(expr_type(double_type(), 2U), true));
  arg_types.push_back(function_arg_type(expr_type(int_type(), 1U)));
  arg_types.push_back(function_arg_type(expr_type(vector_type(), 0U)));
  fs.add(name, return_type, arg_types);

  // check definition
  stan::lang::function_signature_t sig(return_type, arg_types);
  stan::lang::function_signature_t sig2 = fs.get_definition(name, sig);
  EXPECT_EQ(sig, fs.get_definition(name, sig));

  // check function arguments
  EXPECT_TRUE(sig2.second[0].data_only_);
  EXPECT_FALSE(sig2.second[1].data_only_);
  EXPECT_FALSE(sig2.second[2].data_only_);
}

TEST(langAst, discreteFirstArg) {
  // true if first argument to function is always discrete
  EXPECT_TRUE(function_signatures::instance()
              .discrete_first_arg("poisson_log"));
  EXPECT_FALSE(function_signatures::instance()
              .discrete_first_arg("normal_log"));
}

TEST(langAst, printSignature) {
  std::vector<function_arg_type> arg_types;
  arg_types.push_back(function_arg_type(expr_type(double_type(), 2U)));
  arg_types.push_back(function_arg_type(expr_type(int_type(), 1U)));
  arg_types.push_back(function_arg_type(expr_type(vector_type(), 0U)));
  std::string name = "foo";

  std::stringstream platform_eol_ss;
  platform_eol_ss << std::endl;
  std::string platform_eol = platform_eol_ss.str();

  std::stringstream msgs1;
  bool sampling_error_style1 = true;
  stan::lang::print_signature(name, arg_types, sampling_error_style1, msgs1);
  EXPECT_EQ("  real[,] ~ foo(int[], vector)" + platform_eol,
            msgs1.str());

  std::stringstream msgs2;
  bool sampling_error_style2 = false;
  stan::lang::print_signature(name, arg_types, sampling_error_style2, msgs2);
  EXPECT_EQ("  foo(real[,], int[], vector)" + platform_eol,
            msgs2.str());

  arg_types.push_back(function_arg_type(expr_type(matrix_type(), 0U), true));
  arg_types.push_back(function_arg_type(expr_type(matrix_type(), 0U), false));

  std::stringstream msgs3;
  stan::lang::print_signature(name, arg_types, sampling_error_style2, msgs3);
  EXPECT_EQ("  foo(real[,], int[], vector, data matrix, matrix)" + platform_eol,
            msgs3.str());


}

TEST(langAst, hasVar) {
  using stan::lang::base_var_decl;
  using stan::lang::binary_op;
  using stan::lang::expression;
  using stan::lang::model_name_origin;
  using stan::lang::parameter_origin;
  using stan::lang::unary_op;
  using stan::lang::scope;
  using stan::lang::variable;
  using stan::lang::variable_map;

  variable_map vm;
  vector<expression> dims;
  base_var_decl alpha_decl = base_var_decl("alpha",dims,double_type());
  scope alpha_origin = parameter_origin;
  vm.add("alpha", alpha_decl, alpha_origin);

  variable v("alpha");
  v.set_type(double_type(), 2U);
  expression e(v);
  EXPECT_TRUE(has_var(e, vm));

  vm.add("beta",
         base_var_decl("beta", vector<expression>(), int_type()),
         model_name_origin);
  variable v_beta("beta");
  v_beta.set_type(int_type(), 0U);
  expression e_beta(v_beta);
  EXPECT_FALSE(has_var(e_beta, vm));

  expression e2(binary_op(e,"+",e));
  EXPECT_TRUE(has_var(e2,vm));

  expression e_beta2(unary_op('!',unary_op('-',e_beta)));
  EXPECT_FALSE(has_var(e_beta2,vm));
}

TEST(lang_ast,expr_type_num_dims) {
  EXPECT_EQ(0U,expr_type().num_dims());
  EXPECT_EQ(2U,expr_type(int_type(),2U).num_dims());
  EXPECT_EQ(2U,expr_type(vector_type(),2U).num_dims());
}

TEST(lang_ast,expr_type_is_primitive) {
  EXPECT_TRUE(expr_type(double_type()).is_primitive());
  EXPECT_TRUE(expr_type(int_type()).is_primitive());
  EXPECT_FALSE(expr_type(vector_type()).is_primitive());
  EXPECT_FALSE(expr_type(row_vector_type()).is_primitive());
  EXPECT_FALSE(expr_type(matrix_type()).is_primitive());
  EXPECT_FALSE(expr_type(int_type(),2U).is_primitive());
}

TEST(lang_ast,expr_type_is_primitive_int) {
  EXPECT_FALSE(expr_type(double_type()).is_primitive_int());
  EXPECT_TRUE(expr_type(int_type()).is_primitive_int());
  EXPECT_FALSE(expr_type(vector_type()).is_primitive_int());
  EXPECT_FALSE(expr_type(row_vector_type()).is_primitive_int());
  EXPECT_FALSE(expr_type(matrix_type()).is_primitive_int());
  EXPECT_FALSE(expr_type(int_type(),2U).is_primitive_int());
}

TEST(lang_ast,expr_type_is_primitive_double) {
  EXPECT_TRUE(expr_type(double_type()).is_primitive_double());
  EXPECT_FALSE(expr_type(int_type()).is_primitive_double());
  EXPECT_FALSE(expr_type(vector_type()).is_primitive_double());
  EXPECT_FALSE(expr_type(row_vector_type()).is_primitive_double());
  EXPECT_FALSE(expr_type(matrix_type()).is_primitive_double());
  EXPECT_FALSE(expr_type(int_type(),2U).is_primitive_double());
}

TEST(lang_ast,expr_type_eq) {
  EXPECT_EQ(expr_type(double_type()),expr_type(double_type()));
  EXPECT_EQ(expr_type(double_type(),1U),expr_type(double_type(),1U));
  EXPECT_NE(expr_type(int_type()), expr_type(double_type()));
  EXPECT_NE(expr_type(int_type(),1), expr_type(int_type(),2));
  EXPECT_TRUE(expr_type(int_type(),1) != expr_type(int_type(),2));
  EXPECT_FALSE(expr_type(int_type(),1) == expr_type(int_type(),2));
}

TEST(lang_ast,base_expr_type_vis) {
  EXPECT_TRUE(base_expr_type(ill_formed_type()).is_ill_formed_type());
  EXPECT_TRUE(base_expr_type(void_type()).is_void_type());
  EXPECT_TRUE(base_expr_type(int_type()).is_int_type());
  EXPECT_TRUE(base_expr_type(double_type()).is_double_type());
  EXPECT_TRUE(base_expr_type(vector_type()).is_vector_type());
  EXPECT_TRUE(base_expr_type(row_vector_type()).is_row_vector_type());
  EXPECT_TRUE(base_expr_type(matrix_type()).is_matrix_type());

  EXPECT_FALSE(base_expr_type(void_type()).is_ill_formed_type());
  EXPECT_FALSE(base_expr_type(int_type()).is_void_type());
  EXPECT_FALSE(base_expr_type(double_type()).is_int_type());
  EXPECT_FALSE(base_expr_type(int_type()).is_double_type());
  EXPECT_FALSE(base_expr_type(int_type()).is_vector_type());
  EXPECT_FALSE(base_expr_type(vector_type()).is_row_vector_type());
  EXPECT_FALSE(base_expr_type(vector_type()).is_matrix_type());
}


TEST(lang_ast,base_expr_type_compare_ops) {
  EXPECT_TRUE(base_expr_type(int_type())
              == base_expr_type(int_type()));
  EXPECT_TRUE(base_expr_type(int_type())
              != base_expr_type(double_type()));
  EXPECT_FALSE(base_expr_type(int_type())
              != base_expr_type(int_type()));
  EXPECT_TRUE(base_expr_type(int_type())
              >= base_expr_type(int_type()));
  EXPECT_TRUE(base_expr_type(int_type())
              <= base_expr_type(int_type()));
  EXPECT_FALSE(base_expr_type(int_type())
               > base_expr_type(int_type()));
  EXPECT_FALSE(base_expr_type(int_type())
               < base_expr_type(int_type()));
  EXPECT_TRUE(base_expr_type(ill_formed_type())
              < base_expr_type(int_type()));
  EXPECT_TRUE(base_expr_type(void_type())
              < base_expr_type(double_type()));
  EXPECT_TRUE(base_expr_type(ill_formed_type())
              < base_expr_type(double_type()));
  EXPECT_TRUE(base_expr_type(void_type())
              < base_expr_type(vector_type()));
  EXPECT_TRUE(base_expr_type(ill_formed_type())
              < base_expr_type(row_vector_type()));
  EXPECT_TRUE(base_expr_type(void_type())
              < base_expr_type(matrix_type()));

  EXPECT_FALSE(base_expr_type(ill_formed_type())
              < base_expr_type(ill_formed_type()));
  EXPECT_FALSE(base_expr_type(void_type())
              < base_expr_type(void_type()));
  EXPECT_FALSE(base_expr_type(int_type())
               < base_expr_type(int_type()));
  EXPECT_FALSE(base_expr_type(double_type())
               < base_expr_type(double_type()));
  EXPECT_FALSE(base_expr_type(vector_type())
              < base_expr_type(vector_type()));
  EXPECT_FALSE(base_expr_type(row_vector_type())
              < base_expr_type(row_vector_type()));
  EXPECT_FALSE(base_expr_type(matrix_type())
              < base_expr_type(matrix_type()));

  EXPECT_FALSE(base_expr_type(ill_formed_type())
              > base_expr_type(ill_formed_type()));
  EXPECT_FALSE(base_expr_type(void_type())
              > base_expr_type(void_type()));
  EXPECT_FALSE(base_expr_type(int_type())
               > base_expr_type(int_type()));
  EXPECT_FALSE(base_expr_type(double_type())
               > base_expr_type(double_type()));
  EXPECT_FALSE(base_expr_type(vector_type())
              > base_expr_type(vector_type()));
  EXPECT_FALSE(base_expr_type(row_vector_type())
              > base_expr_type(row_vector_type()));
  EXPECT_FALSE(base_expr_type(matrix_type())
              > base_expr_type(matrix_type()));

  EXPECT_FALSE(base_expr_type(ill_formed_type())
              != base_expr_type(ill_formed_type()));
  EXPECT_FALSE(base_expr_type(void_type())
              != base_expr_type(void_type()));
  EXPECT_FALSE(base_expr_type(int_type())
               != base_expr_type(int_type()));
  EXPECT_FALSE(base_expr_type(double_type())
               != base_expr_type(double_type()));
  EXPECT_FALSE(base_expr_type(vector_type())
              != base_expr_type(vector_type()));
  EXPECT_FALSE(base_expr_type(row_vector_type())
              != base_expr_type(row_vector_type()));
  EXPECT_FALSE(base_expr_type(matrix_type())
              != base_expr_type(matrix_type()));
}

TEST(lang_ast,expr_type_type) {
  EXPECT_EQ(base_expr_type(double_type()),
              expr_type(double_type()).type());
  EXPECT_EQ(base_expr_type(double_type()),
            expr_type(double_type(),3U).type());
  EXPECT_NE(base_expr_type(double_type()),
            expr_type(int_type()).type());
  EXPECT_NE(base_expr_type(double_type()),
            expr_type(vector_type(),2U).type());
}

std::vector<expr_type> expr_type_vec() {
  return std::vector<expr_type>();
}

std::vector<expr_type> expr_type_vec(const expr_type& t1) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  return etv;
}
std::vector<expr_type> expr_type_vec(const expr_type& t1,
                                     const expr_type& t2) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  return etv;
}
std::vector<expr_type> expr_type_vec(const expr_type& t1,
                                     const expr_type& t2,
                                     const expr_type& t3) {
  std::vector<expr_type> etv;
  etv.push_back(t1);
  etv.push_back(t2);
  etv.push_back(t3);
  return etv;
}

TEST(lang_ast,function_signatures_log_sum_exp_1) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(double_type(),1U)),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_2) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(double_type()),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(double_type()),
                                             expr_type(double_type())),
                               error_msgs));
}

TEST(lang_ast, function_signatures_add) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;

  EXPECT_EQ(expr_type(double_type()),
            fs.get_result_type("sqrt", expr_type_vec(expr_type(double_type())),
                               error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__", expr_type_vec(), error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__", expr_type_vec(expr_type(double_type())), error_msgs));

  // these next two conflict
  fs.add("bar__", expr_type(double_type()), expr_type(int_type()), expr_type(double_type()));
  fs.add("bar__", expr_type(double_type()), expr_type(double_type()), expr_type(int_type()));
  EXPECT_EQ(expr_type(),
            fs.get_result_type("bar__", expr_type_vec(expr_type(int_type()), expr_type(int_type())),
                               error_msgs));

  // after this, should be resolvable
  fs.add("bar__", expr_type(int_type()), expr_type(int_type()), expr_type(int_type()));
  EXPECT_EQ(expr_type(int_type()),
            fs.get_result_type("bar__", expr_type_vec(expr_type(int_type()), expr_type(int_type())),
                               error_msgs));

}

TEST(langAst, voidType) {
  std::stringstream ss;
  stan::lang::write_base_expr_type(ss, void_type());
  EXPECT_EQ("void", ss.str());
  expr_type et(void_type(), 0);
  EXPECT_TRUE(et.type().is_void_type());
}

TEST(langAst, baseVarDecl) {
  std::vector<stan::lang::expression> dims;
  dims.push_back(stan::lang::expression(stan::lang::int_literal(0)));
  stan::lang::base_var_decl bvd("foo", dims, int_type());
  EXPECT_EQ("foo", bvd.name_);
  EXPECT_EQ(1U, bvd.dims_.size());
  EXPECT_EQ(stan::lang::expression(stan::lang::int_literal(0)).expression_type(),
            bvd.dims_[0].expression_type());
  EXPECT_EQ(base_expr_type(int_type()), bvd.base_type_);
}

TEST(langAst, argDecl) {
  stan::lang::arg_decl ad;
  ad.arg_type_ = expr_type(int_type(), 0);
  ad.name_ = "foo";
  stan::lang::base_var_decl bvd = ad.base_variable_declaration();
  EXPECT_EQ("foo", bvd.name_);
  EXPECT_EQ(0U, bvd.dims_.size());
  EXPECT_EQ(base_expr_type(int_type()), bvd.base_type_);
}

TEST(langAst,functionDeclDef) {
  stan::lang::function_decl_def fdd(expr_type(stan::lang::int_type(), 0),
                                  "foo",
                                  std::vector<stan::lang::arg_decl>(),
                                  stan::lang::statement(stan::lang::no_op_statement()));
  EXPECT_EQ("foo", fdd.name_);
  EXPECT_TRUE(fdd.body_.is_no_op_statement());
  EXPECT_EQ(0U, fdd.arg_decls_.size());
  EXPECT_TRUE(fdd.return_type_.is_primitive_int());
}

TEST(langAst, functionDeclDefs) {
  stan::lang::function_decl_def fdd1(expr_type(stan::lang::int_type(), 0),
                                     "foo",
                                     std::vector<stan::lang::arg_decl>(),
                                     stan::lang::statement(stan::lang::no_op_statement()));
  stan::lang::arg_decl ad;
  ad.arg_type_ = expr_type(int_type(), 0);
  ad.name_ = "foo";
  std::vector<stan::lang::arg_decl> arg_decls;
  arg_decls.push_back(ad);
  stan::lang::function_decl_def fdd2(expr_type(stan::lang::double_type(), 3),
                                     "bar",
                                     arg_decls,
                                     stan::lang::statement(stan::lang::no_op_statement()));
  std::vector<stan::lang::function_decl_def> vec_fdds;
  vec_fdds.push_back(fdd1);
  vec_fdds.push_back(fdd2);
  stan::lang::function_decl_defs fdds(vec_fdds);
  EXPECT_EQ(2U, fdds.decl_defs_.size());
}

TEST(langAst, hasRngSuffix) {
  EXPECT_TRUE(stan::lang::has_rng_suffix("foo_rng"));
  EXPECT_FALSE(stan::lang::has_rng_suffix("foo.rng"));
  EXPECT_FALSE(stan::lang::has_rng_suffix("foo.bar"));
}

TEST(langAst, hasLpSuffix) {
  EXPECT_TRUE(stan::lang::has_lp_suffix("foo_lp"));
  EXPECT_FALSE(stan::lang::has_lp_suffix("foo.lp"));
  EXPECT_FALSE(stan::lang::has_lp_suffix("foo.bar"));
}

TEST(langAst, isUserDefined) {
  using stan::lang::function_signature_t;
  using stan::lang::expr_type;
  using stan::lang::expression;
  using stan::lang::is_user_defined;
  using stan::lang::int_literal;
  using stan::lang::double_literal;
  using std::vector;
  using std::string;
  using std::pair;
  vector<expression> args;
  string name = "foo";
  EXPECT_FALSE(is_user_defined(name, args));
  args.push_back(expression(int_literal(0)));
  EXPECT_FALSE(is_user_defined(name, args));

  vector<function_arg_type> arg_types;
  arg_types.push_back(function_arg_type(expr_type(int_type(),0)));
  expr_type result_type(double_type(),0);
  // must add first, before making user defined
  function_signatures::instance().add(name, result_type, arg_types);
  function_signature_t sig(result_type, arg_types);
  pair<string, function_signature_t> name_sig(name, sig);

  function_signatures::instance().set_user_defined(name_sig);

  EXPECT_TRUE(is_user_defined(name, args));
  EXPECT_TRUE(function_signatures::instance().is_user_defined(name_sig));
  EXPECT_FALSE(is_user_defined_prob_function("foo",
                                             expression(double_literal(1.3)),
                                             args));

  string name_pf = "bar_log";
  pair<string, function_signature_t> name_sig_pf(name_pf, sig);
  function_signatures::instance().add(name_pf, result_type, arg_types);
  function_signatures::instance().set_user_defined(name_sig_pf);

  vector<expression> args_pf;
  EXPECT_TRUE(is_user_defined_prob_function("bar_log",
                                            expression(int_literal(2)), // first arg
                                            args_pf));                  // remaining args
}

TEST(langAst, resetSigs) {
  using std::set;
  using std::string;

  stan::lang::function_signatures::reset_sigs();

  // test can get, destroy, then get
  stan::lang::function_signatures& fs1
    = stan::lang::function_signatures::instance();
  set<string> ks1 = fs1.key_set();
  size_t keyset_size = ks1.size();
  EXPECT_TRUE(keyset_size > 0);

  stan::lang::function_signatures::reset_sigs();

  stan::lang::function_signatures& fs2
    = stan::lang::function_signatures::instance();

  set<string> ks2 = fs2.key_set();
  EXPECT_EQ(keyset_size, ks2.size());
}

TEST(langAst, solveOde) {
  using stan::lang::integrate_ode;
  using stan::lang::variable;
  using stan::lang::expr_type;
  using stan::lang::expression;

  integrate_ode so; // null ctor should work and not raise error

  std::string integration_function_name = "bar";
  std::string system_function_name = "foo";

  variable y0("y0_var_name");
  y0.set_type(double_type(), 1);  // plain old vector

  variable t0("t0_var_name");
  t0.set_type(double_type(), 0);  // double

  variable ts("ts_var_name");
  ts.set_type(double_type(), 1);

  variable theta("theta_var_name");
  theta.set_type(double_type(), 1);

  variable x("x_var_name");
  x.set_type(double_type(), 1);

  variable x_int("x_int_var_name");
  x.set_type(int_type(), 1);

  // example of instantiation
  integrate_ode so2(integration_function_name, system_function_name,
                    y0, t0, ts, theta, x, x_int);

  // dumb test to make sure we at least get the right types back
  EXPECT_EQ(integration_function_name, so2.integration_function_name_);
  EXPECT_EQ(system_function_name, so2.system_function_name_);
  EXPECT_EQ(y0.type_, so2.y0_.expression_type());
  EXPECT_EQ(t0.type_, so2.t0_.expression_type());
  EXPECT_EQ(ts.type_, so2.ts_.expression_type());
  EXPECT_EQ(theta.type_, so2.theta_.expression_type());
  EXPECT_EQ(x.type_, so2.x_.expression_type());
  EXPECT_EQ(x_int.type_, so2.x_int_.expression_type());

  expression e2(so2);
  EXPECT_EQ(expr_type(double_type(),2), e2.expression_type());
}

TEST(langAst, solveAlgebra) {
    using stan::lang::algebra_solver;
    using stan::lang::variable;
    using stan::lang::expr_type;
    using stan::lang::expression;

    algebra_solver so;  // null ctor should work and not raise error
    std::string system_function_name = "bronzino";

    variable y("y_var_name");
    y.set_type(vector_type(), 0);  // vector from Eigen

    variable theta("theta_var_name");
    theta.set_type(vector_type(), 0);

    variable x_r("x_r_r_var_name");
    x_r.set_type(double_type(), 1);  // plain old vector

    variable x_i("x_i_var_name");
    x_i.set_type(int_type(), 1);

    // example of instantiation
    algebra_solver so2(system_function_name, y, theta, x_r, x_i);

    // dumb test to make sure we at least get the right types back
    EXPECT_EQ(system_function_name, so2.system_function_name_);
    EXPECT_EQ(y.type_, so2.y_.expression_type());
    EXPECT_EQ(theta.type_, so2.theta_.expression_type());
    EXPECT_EQ(x_r.type_, so2.x_r_.expression_type());
    EXPECT_EQ(x_i.type_, so2.x_i_.expression_type());

    expression e2(so2);
    EXPECT_EQ(expr_type(vector_type(), 0), e2.expression_type());
}

void testTotalDims(int expected_total_dims,
                   const stan::lang::base_expr_type& base_type,
                   size_t num_dims) {
  using stan::lang::expression;
  using stan::lang::variable;

  variable v("foo");
  v.set_type(base_type, num_dims);

  expression e(v);
  EXPECT_EQ(expected_total_dims, e.total_dims());
}

TEST(gmAst,expressionTotalDims) {
  testTotalDims(0, double_type(), 0);
  testTotalDims(2, double_type(), 2);
  testTotalDims(0, int_type(), 0);
  testTotalDims(2, int_type(), 2);
  testTotalDims(2, matrix_type(), 0);
  testTotalDims(5, matrix_type(), 3);
  testTotalDims(1, vector_type(), 0);
  testTotalDims(4, vector_type(), 3);
  testTotalDims(1, row_vector_type(), 0);
  testTotalDims(4, row_vector_type(), 3);
}

TEST(gmAst, isBinaryOperator) {
  using stan::lang::is_binary_operator;
  EXPECT_TRUE(is_binary_operator("add"));
  EXPECT_TRUE(is_binary_operator("elt_divide"));
  EXPECT_FALSE(is_binary_operator("foo"));
}

TEST(gmAst, isUnaryOperator) {
  using stan::lang::is_unary_operator;
  EXPECT_TRUE(is_unary_operator("minus"));
  EXPECT_FALSE(is_unary_operator("foo"));
}

TEST(gmAst, isUnaryPostfix) {
  using stan::lang::is_unary_postfix_operator;
  EXPECT_TRUE(is_unary_postfix_operator("transpose"));
  EXPECT_FALSE(is_unary_postfix_operator("bar"));
}

TEST(gmAst, funNameToOperator) {
  using stan::lang::fun_name_to_operator;
  EXPECT_EQ("+", fun_name_to_operator("add"));
  EXPECT_EQ("-", fun_name_to_operator("minus"));
  EXPECT_EQ("ERROR", fun_name_to_operator("foo"));
}

TEST(langAst, uniIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::uni_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(base_expr_type(int_type()), i.idx_.expression_type().type());
  EXPECT_EQ(0, i.idx_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, multiIdx) {
  stan::lang::variable v("foo");
  v.set_type(int_type(), 1);
  stan::lang::expression e(v);
  stan::lang::multi_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(base_expr_type(int_type()), i.idxs_.expression_type().type());
  EXPECT_EQ(1, i.idxs_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, omniIdx) {
  // nothing to store or retrieve for omni
  EXPECT_NO_THROW(stan::lang::omni_idx());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, lbIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::lb_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(base_expr_type(int_type()), i.lb_.expression_type().type());
  EXPECT_EQ(0, i.lb_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, ubIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::ub_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(base_expr_type(int_type()), i.ub_.expression_type().type());
  EXPECT_EQ(0, i.ub_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, lubIdx) {
  stan::lang::expression e1(stan::lang::int_literal(3));
  stan::lang::variable v("foo");
  v.set_type(int_type(), 0);
  stan::lang::expression e2(v);
  stan::lang::lub_idx i(e1,e2);
  // test proper type storage and retrieval
  EXPECT_EQ(base_expr_type(int_type()), i.lb_.expression_type().type());
  EXPECT_EQ(0, i.lb_.expression_type().num_dims());
  EXPECT_EQ(base_expr_type(int_type()), i.ub_.expression_type().type());
  EXPECT_EQ(0, i.ub_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}

TEST(langAst, assgn) {
  stan::lang::variable v("foo");
  v.set_type(double_type(), 0);
  std::vector<stan::lang::idx> is;
  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui(e_int3);
  stan::lang::idx idx0(ui);
  is.push_back(idx0);
  std::string op("=");
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::assgn a(v, is, op, e);
  // retrieve indexes
  EXPECT_EQ(1, a.idxs_.size());
  // retrieve LHS variable
  EXPECT_EQ(0, a.lhs_var_.type_.num_dims());
  EXPECT_EQ(base_expr_type(double_type()), a.lhs_var_.type_.type());
  // retrieve RHS expression
  EXPECT_EQ(0, a.rhs_.expression_type().num_dims());
  EXPECT_EQ(base_expr_type(int_type()), a.rhs_.expression_type().type());
}

// Type Inference Tests for Generalized Indexing

// tests recovery of base expression type and number of dims
// given expression and indexing
void test_recover(base_expr_type base_et_expected,
                  size_t num_dims_expected,
                  base_expr_type base_et, size_t num_dims,
                  const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  v.set_type(base_et, num_dims);
  stan::lang::expression e(v);
  expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(base_et_expected, et.base_type_);
  EXPECT_EQ(num_dims_expected, et.num_dims_);
}

void test_err(base_expr_type base_et, size_t num_dims,
              const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  v.set_type(base_et, num_dims);
  stan::lang::expression e(v);
  expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(base_expr_type(ill_formed_type()), et.base_type_);
}

TEST(langAst, idxs) {
  const stan::lang::base_expr_type bet[]
    = { base_expr_type(int_type()), base_expr_type(double_type()),
        base_expr_type(vector_type()), base_expr_type(row_vector_type()),
        base_expr_type(matrix_type()) };
  vector<idx> idxs;
  for (size_t n = 0; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n, bet[i], n, idxs);
}

void one_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[]
    = { base_expr_type(int_type()), base_expr_type(double_type()),
        base_expr_type(vector_type()), base_expr_type(row_vector_type()),
        base_expr_type(matrix_type()) };
  for (size_t n = 1; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void one_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(base_expr_type(double_type()), 0U, idxs);
  test_err(base_expr_type(int_type()), 0U, idxs);
}

TEST(langAst, idxs0) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));

  one_index_errs(idxs);
  one_index_recover(idxs, 1U);
  test_recover(base_expr_type(double_type()), 0U,
               base_expr_type(vector_type()), 0U, idxs);
  test_recover(base_expr_type(double_type()), 0U,
               base_expr_type(row_vector_type()), 0U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U,
               base_expr_type(matrix_type()), 0U, idxs);
}

TEST(langAst, idxs1) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());

  one_index_errs(idxs);
  one_index_recover(idxs, 0U);
  test_recover(base_expr_type(vector_type()), 0U, base_expr_type(vector_type()), 0U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(row_vector_type()), 0U, idxs);
  test_recover(base_expr_type(matrix_type()), 0U, base_expr_type(matrix_type()), 0U, idxs);
}

void two_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[]
    = { base_expr_type(int_type()), base_expr_type(double_type()),
        base_expr_type(vector_type()), base_expr_type(row_vector_type()),
        base_expr_type(base_expr_type(matrix_type())) };
  for (size_t n = 2; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void two_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(base_expr_type(double_type()), 0U, idxs);
  test_err(base_expr_type(double_type()), 1U, idxs);
  test_err(base_expr_type(int_type()), 0U, idxs);
  test_err(base_expr_type(int_type()), 1U, idxs);
  test_err(base_expr_type(vector_type()), 0U, idxs);
  test_err(base_expr_type(row_vector_type()), 0U, idxs);
}

TEST(langAst, idxs00) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 2U);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(vector_type()), 1U, idxs);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(row_vector_type()), 1U, idxs);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(matrix_type()), 0U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs01) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(base_expr_type(vector_type()), 0U, base_expr_type(vector_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(row_vector_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(matrix_type()), 0U, idxs);
  test_recover(base_expr_type(matrix_type()), 0U, base_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs10) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(base_expr_type(double_type()), 1U, base_expr_type(vector_type()), 1U, idxs);
  test_recover(base_expr_type(double_type()), 1U, base_expr_type(row_vector_type()), 1U, idxs);
  test_recover(base_expr_type(vector_type()), 0U, base_expr_type(matrix_type()), 0U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
}

TEST(langAst, idxs11) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 0U);
  test_recover(base_expr_type(vector_type()), 1U, base_expr_type(vector_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(row_vector_type()), 1U, idxs);
  test_recover(base_expr_type(matrix_type()), 0U, base_expr_type(matrix_type()), 0U, idxs);
  test_recover(base_expr_type(matrix_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
}

void three_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[]
    = { base_expr_type(int_type()), base_expr_type(double_type()),
        base_expr_type(vector_type()), base_expr_type(row_vector_type()),
        base_expr_type(matrix_type()) };
  for (int i = 0; i < 5; ++i)
    for (size_t n = 3; n < 5; ++n)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}

void three_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(base_expr_type(double_type()), 0U, idxs);
  test_err(base_expr_type(double_type()), 1U, idxs);
  test_err(base_expr_type(double_type()), 2U, idxs);
  test_err(base_expr_type(int_type()), 0U, idxs);
  test_err(base_expr_type(int_type()), 1U, idxs);
  test_err(base_expr_type(int_type()), 2U, idxs);
  test_err(base_expr_type(vector_type()), 0U, idxs);
  test_err(base_expr_type(vector_type()), 1U, idxs);
  test_err(base_expr_type(row_vector_type()), 0U, idxs);
  test_err(base_expr_type(row_vector_type()), 1U, idxs);
  test_err(base_expr_type(matrix_type()), 0U, idxs);
}

TEST(langAst, idxs000) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(uni_idx(expression(int_literal(7))));

  three_index_errs(idxs);
  three_index_recover(idxs, 3U);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(double_type()), 0U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs001) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(base_expr_type(vector_type()), 0U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 0U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(matrix_type()), 0U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs011) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(base_expr_type(vector_type()), 1U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(matrix_type()), 0U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(matrix_type()), 1U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs100) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(base_expr_type(double_type()), 1U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(double_type()), 1U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(double_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs101) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(base_expr_type(vector_type()), 1U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(matrix_type()), 1U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs110) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(base_expr_type(double_type()), 2U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(double_type()), 2U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(vector_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(row_vector_type()), 2U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, idxs111) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 0U);
  test_recover(base_expr_type(vector_type()), 2U, base_expr_type(vector_type()), 2U, idxs);
  test_recover(base_expr_type(row_vector_type()), 2U, base_expr_type(row_vector_type()), 2U, idxs);
  test_recover(base_expr_type(matrix_type()), 1U, base_expr_type(matrix_type()), 1U, idxs);
  test_recover(base_expr_type(matrix_type()), 2U, base_expr_type(matrix_type()), 2U, idxs);
}

TEST(langAst, indexOpSliced) {
  using stan::lang::index_op_sliced;

  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  // no need to retest all of type inference here --- just that it's plumbed
  index_op_sliced ios;
  stan::lang::variable v("foo");
  v.set_type(vector_type(), 1U);
  ios.expr_ = v;
  ios.idxs_ = idxs;
  ios.infer_type();
  EXPECT_EQ(base_expr_type(double_type()), ios.type_.base_type_);
  EXPECT_EQ(1U, ios.type_.num_dims_);
}

TEST(langAst, lhsVarOccursOnRhs) {
  stan::lang::variable v("foo");
  v.set_type(double_type(), 0);
  std::vector<stan::lang::idx> is;
  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui(e_int3);
  stan::lang::idx idx0(ui);
  is.push_back(idx0);
  std::string op("=");
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::assgn a(v, is, op, e);
  EXPECT_FALSE(a.lhs_var_occurs_on_rhs());

  std::vector<stan::lang::idx> is2;
  stan::lang::assgn a2(v, is2, op, v);
  EXPECT_TRUE(a2.lhs_var_occurs_on_rhs());

  stan::lang::unary_op uo('+', v);
  stan::lang::assgn a3(v, is2, op, uo);
  EXPECT_TRUE(a3.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo(v, "-", e_int3);
  stan::lang::assgn a4(v, is2, op, bo);
  EXPECT_TRUE(a4.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo2(e_int3, "*", e_int3);
  stan::lang::assgn a5(v, is2, op, bo2);
  EXPECT_FALSE(a5.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo3(e_int3, "*", bo);
  stan::lang::assgn a6(v, is2, op, bo3);
  EXPECT_TRUE(a6.lhs_var_occurs_on_rhs());

  stan::lang::index_op_sliced ios(v, is2);
  stan::lang::assgn a7(v, is2, op, ios);
  EXPECT_TRUE(a7.lhs_var_occurs_on_rhs());
}

TEST(StanLangAstFun, is_space) {
  using stan::lang::is_space;
  EXPECT_TRUE(is_space(' '));
  EXPECT_TRUE(is_space('\n'));
  EXPECT_TRUE(is_space('\t'));
  EXPECT_TRUE(is_space('\r'));

  EXPECT_FALSE(is_space('a'));
  EXPECT_FALSE(is_space('2'));
}

TEST(StanLangAstFun, is_nonempty) {
  using stan::lang::is_nonempty;
  EXPECT_FALSE(is_nonempty(" "));
  EXPECT_FALSE(is_nonempty("\n"));
  EXPECT_FALSE(is_nonempty("\t"));
  EXPECT_FALSE(is_nonempty("\r"));
  EXPECT_FALSE(is_nonempty("   \r  \t  "));

  EXPECT_TRUE(is_nonempty("1"));
  EXPECT_TRUE(is_nonempty("  \r\n \n 1  \n"));
}

template <typename T>
void expect_has_var_bool(const T& x) {
  EXPECT_TRUE(x.has_var_ == 0 || x.has_var_ == 1);
}


TEST(StanLangAst, ConditionalOp) {
  expect_has_var_bool(stan::lang::conditional_op());

  stan::lang::expression e = int_literal(3);
  expect_has_var_bool(stan::lang::conditional_op(e, e, e));
}

TEST(StanLangAst, RowVectorExpr) {
  expect_has_var_bool(stan::lang::row_vector_expr());
}

TEST(StanLangAst, MatrixExpr) {
  expect_has_var_bool(stan::lang::matrix_expr());
}

TEST(StanLangAst, Sample) {
  stan::lang::sample s;
  EXPECT_TRUE(s.is_discrete_ == true || s.is_discrete_ == false);

  stan::lang::expression e = int_literal(3);
  stan::lang::distribution d;
  stan::lang::sample s2(e, d);
  EXPECT_TRUE(s2.is_discrete_ == true || s2.is_discrete_ == false);
}

TEST(StanLangAst, Scope) {
  stan::lang::scope s;
  EXPECT_TRUE(s.is_local() == true || s.is_local() == false);

  stan::lang::scope s2(stan::lang::data_origin);
  EXPECT_TRUE(s2.is_local() == true || s2.is_local() == false);
}

TEST(StanLangAst, MapRect) {
  // make sure nullary ctor works
  stan::lang::map_rect mr1;
  EXPECT_TRUE(mr1.call_id_ == -1);

  // test fidelity of storage
  std::string name = "foo";
  stan::lang::expression e1 = int_literal(1);
  stan::lang::expression e2 = int_literal(2);
  stan::lang::expression e3 = int_literal(3);
  stan::lang::expression e4 = int_literal(4);
  stan::lang::map_rect mr(name, e1, e2, e3, e4);
  EXPECT_TRUE(mr.fun_name_ == "foo");
  int_literal lit1 = boost::polymorphic_get<int_literal>(mr.shared_params_.expr_);
  EXPECT_EQ(1, lit1.val_);
  int_literal lit2 = boost::polymorphic_get<int_literal>(mr.job_params_.expr_);
  EXPECT_EQ(2, lit2.val_);
  int_literal lit3 = boost::polymorphic_get<int_literal>(mr.job_data_r_.expr_);
  EXPECT_EQ(3, lit3.val_);
  int_literal lit4 = boost::polymorphic_get<int_literal>(mr.job_data_i_.expr_);
  EXPECT_EQ(4, lit4.val_);
}
