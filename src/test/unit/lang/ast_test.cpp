#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <gtest/gtest.h>
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
using stan::lang::expr_type;
using stan::lang::DOUBLE_T;
using stan::lang::INT_T;
using stan::lang::VECTOR_T;
using stan::lang::ROW_VECTOR_T;
using stan::lang::MATRIX_T;
using std::vector;

TEST(langAst, discreteFirstArg) {
  // true if first argument to function is always discrete
  EXPECT_TRUE(function_signatures::instance()
              .discrete_first_arg("poisson_log"));
  EXPECT_FALSE(function_signatures::instance()
              .discrete_first_arg("normal_log"));
}

TEST(langAst, printSignature) {
  std::vector<expr_type> arg_types;
  arg_types.push_back(expr_type(DOUBLE_T, 2U));
  arg_types.push_back(expr_type(INT_T, 1U));
  arg_types.push_back(expr_type(VECTOR_T, 0U));
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
  base_var_decl alpha_decl = base_var_decl("alpha",dims,DOUBLE_T);
  scope alpha_origin = parameter_origin;
  vm.add("alpha", alpha_decl, alpha_origin);
  
  variable v("alpha");
  v.set_type(DOUBLE_T, 2U);
  expression e(v);
  EXPECT_TRUE(has_var(e, vm));

  vm.add("beta", 
         base_var_decl("beta", vector<expression>(), INT_T),
         model_name_origin);
  variable v_beta("beta");
  v_beta.set_type(INT_T, 0U);
  expression e_beta(v_beta);
  EXPECT_FALSE(has_var(e_beta, vm));

  expression e2(binary_op(e,"+",e));
  EXPECT_TRUE(has_var(e2,vm));

  expression e_beta2(unary_op('!',unary_op('-',e_beta)));
  EXPECT_FALSE(has_var(e_beta2,vm));
}

TEST(lang_ast,expr_type_num_dims) {
  EXPECT_EQ(0U,expr_type().num_dims());
  EXPECT_EQ(2U,expr_type(INT_T,2U).num_dims());
  EXPECT_EQ(2U,expr_type(VECTOR_T,2U).num_dims());
}
TEST(lang_ast,expr_type_is_primitive) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive());
  EXPECT_TRUE(expr_type(INT_T).is_primitive());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive());
}
TEST(lang_ast,expr_type_is_primitive_int) {
  EXPECT_FALSE(expr_type(DOUBLE_T).is_primitive_int());
  EXPECT_TRUE(expr_type(INT_T).is_primitive_int());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_int());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_int());
}
TEST(lang_ast,expr_type_is_primitive_double) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T).is_primitive_double());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_double());
}
TEST(lang_ast,expr_type_eq) {
  EXPECT_EQ(expr_type(DOUBLE_T),expr_type(DOUBLE_T));
  EXPECT_EQ(expr_type(DOUBLE_T,1U),expr_type(DOUBLE_T,1U));
  EXPECT_NE(expr_type(INT_T), expr_type(DOUBLE_T));
  EXPECT_NE(expr_type(INT_T,1), expr_type(INT_T,2));
}
TEST(lang_ast,expr_type_type) {
  EXPECT_EQ(DOUBLE_T,expr_type(DOUBLE_T).type());
  EXPECT_EQ(DOUBLE_T,expr_type(DOUBLE_T,3U).type());
  EXPECT_NE(DOUBLE_T,expr_type(INT_T).type());
  EXPECT_NE(DOUBLE_T,expr_type(VECTOR_T,2U).type());
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
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T,1U)),
                               error_msgs));
}

TEST(lang_ast,function_signatures_log_sum_exp_2) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T),
                                             expr_type(DOUBLE_T)),
                               error_msgs));
}

TEST(lang_ast,function_signatures_add) {
  stan::lang::function_signatures& fs = stan::lang::function_signatures::instance();
  std::stringstream error_msgs;

  EXPECT_EQ(expr_type(DOUBLE_T), 
            fs.get_result_type("sqrt",expr_type_vec(expr_type(DOUBLE_T)),
                               error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__",expr_type_vec(),error_msgs));
  EXPECT_EQ(expr_type(), fs.get_result_type("foo__",expr_type_vec(expr_type(DOUBLE_T)),error_msgs));

  // these next two conflict
  fs.add("bar__",expr_type(DOUBLE_T),expr_type(INT_T),expr_type(DOUBLE_T));
  fs.add("bar__",expr_type(DOUBLE_T),expr_type(DOUBLE_T),expr_type(INT_T));
  EXPECT_EQ(expr_type(), 
            fs.get_result_type("bar__",expr_type_vec(expr_type(INT_T),expr_type(INT_T)),
                               error_msgs));

  // after this, should be resolvable
  fs.add("bar__",expr_type(INT_T), expr_type(INT_T), expr_type(INT_T));
  EXPECT_EQ(expr_type(INT_T), 
            fs.get_result_type("bar__",expr_type_vec(INT_T,INT_T),
                               error_msgs)); // expr_type(INT_T),expr_type(INT_T))));
  
}

TEST(langAst,voidType) {
  EXPECT_EQ(stan::lang::VOID_T, 0);
  std::stringstream ss;
  stan::lang::write_base_expr_type(ss, stan::lang::VOID_T);
  EXPECT_EQ("void",ss.str());

  expr_type et(stan::lang::VOID_T,0);
  EXPECT_TRUE(et.is_void());
}

TEST(langAst,baseVarDecl) {
  std::vector<stan::lang::expression> dims;
  dims.push_back(stan::lang::expression(stan::lang::int_literal(0)));
  stan::lang::base_var_decl bvd("foo", dims, INT_T);
  EXPECT_EQ("foo",bvd.name_);
  EXPECT_EQ(1U, bvd.dims_.size());
  EXPECT_EQ(stan::lang::expression(stan::lang::int_literal(0)).expression_type(),
            bvd.dims_[0].expression_type());
  EXPECT_EQ(INT_T, bvd.base_type_);
}

TEST(langAst,argDecl) {
  stan::lang::arg_decl ad;
  ad.arg_type_ = expr_type(INT_T,0);
  ad.name_ = "foo";
  stan::lang::base_var_decl bvd = ad.base_variable_declaration();
  EXPECT_EQ("foo", bvd.name_);
  EXPECT_EQ(0U, bvd.dims_.size());
  EXPECT_EQ(INT_T, bvd.base_type_);
}

TEST(langAst,functionDeclDef) {
  stan::lang::function_decl_def fdd(expr_type(stan::lang::INT_T,0),
                                  "foo",
                                  std::vector<stan::lang::arg_decl>(),
                                  stan::lang::statement(stan::lang::no_op_statement()));
  EXPECT_EQ("foo",fdd.name_);
  EXPECT_TRUE(fdd.body_.is_no_op_statement());
  EXPECT_EQ(0U,fdd.arg_decls_.size());
  EXPECT_TRUE(fdd.return_type_.is_primitive_int());
}
TEST(langAst,functionDeclDefs) {
  stan::lang::function_decl_def fdd1(expr_type(stan::lang::INT_T,0),
                                  "foo",
                                  std::vector<stan::lang::arg_decl>(),
                                  stan::lang::statement(stan::lang::no_op_statement()));
  stan::lang::arg_decl ad;
  ad.arg_type_ = expr_type(INT_T,0);
  ad.name_ = "foo";
  std::vector<stan::lang::arg_decl> arg_decls;
  arg_decls.push_back(ad);
  stan::lang::function_decl_def fdd2(expr_type(stan::lang::DOUBLE_T,3),
                                  "bar",
                                   arg_decls,
                                   stan::lang::statement(stan::lang::no_op_statement()));
  std::vector<stan::lang::function_decl_def> vec_fdds;
  vec_fdds.push_back(fdd1);
  vec_fdds.push_back(fdd2);
  stan::lang::function_decl_defs fdds(vec_fdds);
  EXPECT_EQ(2U,fdds.decl_defs_.size());
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
  EXPECT_FALSE(is_user_defined(name,args));
  args.push_back(expression(int_literal(0)));
  EXPECT_FALSE(is_user_defined(name,args));

  vector<expr_type> arg_types;
  arg_types.push_back(expr_type(INT_T,0));
  expr_type result_type(DOUBLE_T,0);
  // must add first, before making user defined
  function_signatures::instance().add(name, result_type, arg_types);
  function_signature_t sig(result_type, arg_types);
  pair<string,function_signature_t> name_sig(name,sig);

  function_signatures::instance().set_user_defined(name_sig);
  
  EXPECT_TRUE(is_user_defined(name,args));

  
  EXPECT_TRUE(function_signatures::instance().is_user_defined(name_sig));
                           
  EXPECT_FALSE(is_user_defined_prob_function("foo",
                                             expression(double_literal(1.3)),
                                             args));

  string name_pf = "bar_log";
  pair<string,function_signature_t> name_sig_pf(name_pf,sig);
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
  y0.set_type(DOUBLE_T, 1);  // plain old vector

  variable t0("t0_var_name");
  t0.set_type(DOUBLE_T, 0);  // double

  variable ts("ts_var_name");
  ts.set_type(DOUBLE_T, 1); 

  variable theta("theta_var_name");
  theta.set_type(DOUBLE_T, 1);
  
  variable x("x_var_name");
  x.set_type(DOUBLE_T, 1);

  variable x_int("x_int_var_name");
  x.set_type(INT_T, 1);


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
  EXPECT_EQ(expr_type(DOUBLE_T,2), e2.expression_type());
}

void testTotalDims(int expected_total_dims,
                   const stan::lang::base_expr_type& base_type,
                   size_t num_dims) {
  using stan::lang::expression;
  using stan::lang::variable;

  variable v("foo");
  v.set_type(base_type, num_dims);
  
  expression e(v);
  EXPECT_EQ(expected_total_dims,e.total_dims());
}

TEST(gmAst,expressionTotalDims) {
  testTotalDims(0, DOUBLE_T, 0);
  testTotalDims(2, DOUBLE_T, 2);
  testTotalDims(0, INT_T, 0);
  testTotalDims(2, INT_T, 2);
  testTotalDims(2, MATRIX_T, 0);
  testTotalDims(5, MATRIX_T, 3);
  testTotalDims(1, VECTOR_T, 0);
  testTotalDims(4, VECTOR_T, 3);
  testTotalDims(1, ROW_VECTOR_T, 0);
  testTotalDims(4, ROW_VECTOR_T, 3);
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
  EXPECT_EQ(INT_T, i.idx_.expression_type().type());
  EXPECT_EQ(0, i.idx_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}
TEST(langAst, multiIdx) {
  stan::lang::variable v("foo");
  v.set_type(INT_T, 1);
  stan::lang::expression e(v);
  stan::lang::multi_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(INT_T, i.idxs_.expression_type().type());
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
  EXPECT_EQ(INT_T, i.lb_.expression_type().type());
  EXPECT_EQ(0, i.lb_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}
TEST(langAst, ubIdx) {
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::ub_idx i(e);
  // test proper type storage and retrieval
  EXPECT_EQ(INT_T, i.ub_.expression_type().type());
  EXPECT_EQ(0, i.ub_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}
TEST(langAst, lubIdx) {
  stan::lang::expression e1(stan::lang::int_literal(3));
  stan::lang::variable v("foo");
  v.set_type(INT_T, 0);
  stan::lang::expression e2(v);
  stan::lang::lub_idx i(e1,e2);
  // test proper type storage and retrieval
  EXPECT_EQ(INT_T, i.lb_.expression_type().type());
  EXPECT_EQ(0, i.lb_.expression_type().num_dims());
  EXPECT_EQ(INT_T, i.ub_.expression_type().type());
  EXPECT_EQ(0, i.ub_.expression_type().num_dims());
  // test allow construction
  EXPECT_NO_THROW(stan::lang::idx(i));
}
TEST(langAst, assgn) {
  stan::lang::variable v("foo");
  v.set_type(DOUBLE_T, 0);
  std::vector<stan::lang::idx> is;
  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui(e_int3);
  stan::lang::idx idx0(ui);
  is.push_back(idx0);
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::assgn a(v, is, e);
  // retrieve indexes
  EXPECT_EQ(1, a.idxs_.size());
  // retrieve LHS variable
  EXPECT_EQ(0, a.lhs_var_.type_.num_dims());
  EXPECT_EQ(DOUBLE_T, a.lhs_var_.type_.type());
  // retrieve RHS expression
  EXPECT_EQ(0, a.rhs_.expression_type().num_dims());
  EXPECT_EQ(INT_T, a.rhs_.expression_type().type());
}

// Type Inference Tests for Generalized Indexing

// tests recovery of base expression type and number of dims
// given expression and indexing
void test_recover(stan::lang::base_expr_type base_et_expected, 
                  size_t num_dims_expected,
                  stan::lang::base_expr_type base_et, size_t num_dims,
                  const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  v.set_type(base_et, num_dims);
  stan::lang::expression e(v);
  stan::lang::expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(base_et_expected, et.base_type_);
  EXPECT_EQ(num_dims_expected, et.num_dims_);
}
void test_err(stan::lang::base_expr_type base_et, size_t num_dims,
              const std::vector<stan::lang::idx>& idxs) {
  stan::lang::variable v("foo");
  v.set_type(base_et, num_dims);
  stan::lang::expression e(v);
  stan::lang::expr_type et = indexed_type(e, idxs);
  EXPECT_EQ(stan::lang::ILL_FORMED_T, et.base_type_);
}

TEST(langAst, idxs) {
  const stan::lang::base_expr_type bet[] 
    = { INT_T, DOUBLE_T, VECTOR_T, ROW_VECTOR_T, MATRIX_T };
  vector<idx> idxs;
  for (size_t n = 0; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n, bet[i], n, idxs);
}

void one_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[] 
    = { INT_T, DOUBLE_T, VECTOR_T, ROW_VECTOR_T, MATRIX_T };
  for (size_t n = 1; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}
void one_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(DOUBLE_T, 0U, idxs);
  test_err(INT_T, 0U, idxs);
}
TEST(langAst, idxs0) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));

  one_index_errs(idxs);
  one_index_recover(idxs, 1U);
  test_recover(DOUBLE_T, 0U, VECTOR_T, 0U, idxs);
  test_recover(DOUBLE_T, 0U, ROW_VECTOR_T, 0U, idxs);
  test_recover(ROW_VECTOR_T, 0U, MATRIX_T, 0U, idxs);
}
TEST(langAst, idxs1) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  
  one_index_errs(idxs);
  one_index_recover(idxs, 0U);
  test_recover(VECTOR_T, 0U, VECTOR_T, 0U, idxs);
  test_recover(ROW_VECTOR_T, 0U, ROW_VECTOR_T, 0U, idxs);
  test_recover(MATRIX_T, 0U, MATRIX_T, 0U, idxs);
}

void two_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[] 
    = { INT_T, DOUBLE_T, VECTOR_T, ROW_VECTOR_T, MATRIX_T };
  for (size_t n = 2; n < 4; ++n)
    for (int i = 0; i < 5; ++i)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}
void two_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(DOUBLE_T, 0U, idxs);
  test_err(DOUBLE_T, 1U, idxs);
  test_err(INT_T, 0U, idxs);
  test_err(INT_T, 1U, idxs);
  test_err(VECTOR_T, 0U, idxs);
  test_err(ROW_VECTOR_T, 0U, idxs);
}
TEST(langAst, idxs00) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 2U);
  test_recover(DOUBLE_T, 0U, VECTOR_T, 1U, idxs);
  test_recover(DOUBLE_T, 0U, ROW_VECTOR_T, 1U, idxs);
  test_recover(DOUBLE_T, 0U, MATRIX_T, 0U, idxs);
  test_recover(ROW_VECTOR_T, 0U, MATRIX_T, 1U, idxs);
}
TEST(langAst, idxs01) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(VECTOR_T, 0U, VECTOR_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 0U, ROW_VECTOR_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 0U, MATRIX_T, 0U, idxs);
  test_recover(MATRIX_T, 0U, MATRIX_T, 1U, idxs);
}
TEST(langAst, idxs10) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(5))));

  two_index_errs(idxs);
  two_index_recover(idxs, 1U);
  test_recover(DOUBLE_T, 1U, VECTOR_T, 1U, idxs);
  test_recover(DOUBLE_T, 1U, ROW_VECTOR_T, 1U, idxs);
  test_recover(VECTOR_T, 0U, MATRIX_T, 0U, idxs);
  test_recover(ROW_VECTOR_T, 1U, MATRIX_T, 1U, idxs);
}
TEST(langAst, idxs11) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  two_index_errs(idxs);
  two_index_recover(idxs, 0U);
  test_recover(VECTOR_T, 1U, VECTOR_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 1U, ROW_VECTOR_T, 1U, idxs);
  test_recover(MATRIX_T, 0U, MATRIX_T, 0U, idxs);
  test_recover(MATRIX_T, 1U, MATRIX_T, 1U, idxs);
}

void three_index_recover(const std::vector<stan::lang::idx>& idxs, size_t redux) {
  const stan::lang::base_expr_type bet[] 
    = { INT_T, DOUBLE_T, VECTOR_T, ROW_VECTOR_T, MATRIX_T };
  for (int i = 0; i < 5; ++i)
    for (size_t n = 3; n < 5; ++n)
      test_recover(bet[i], n - redux, bet[i], n, idxs);
}
void three_index_errs(const std::vector<stan::lang::idx>& idxs) {
  test_err(DOUBLE_T, 0U, idxs);
  test_err(DOUBLE_T, 1U, idxs);
  test_err(DOUBLE_T, 2U, idxs);
  test_err(INT_T, 0U, idxs);
  test_err(INT_T, 1U, idxs);
  test_err(INT_T, 2U, idxs);
  test_err(VECTOR_T, 0U, idxs);
  test_err(VECTOR_T, 1U, idxs);
  test_err(ROW_VECTOR_T, 0U, idxs);
  test_err(ROW_VECTOR_T, 1U, idxs);
  test_err(MATRIX_T, 0U, idxs);
}
TEST(langAst, idxs000) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(uni_idx(expression(int_literal(7))));

  three_index_errs(idxs);
  three_index_recover(idxs, 3U);
  test_recover(DOUBLE_T, 0U, VECTOR_T, 2U, idxs);
  test_recover(DOUBLE_T, 0U, ROW_VECTOR_T, 2U, idxs);
  test_recover(DOUBLE_T, 0U, MATRIX_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 0U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs001) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(VECTOR_T, 0U, VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 0U, ROW_VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 0U, MATRIX_T, 1U, idxs);
  test_recover(MATRIX_T, 0U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs011) {
  vector<idx> idxs;
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(VECTOR_T, 1U, VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 1U, ROW_VECTOR_T, 2U, idxs);
  test_recover(MATRIX_T, 0U, MATRIX_T, 1U, idxs);
  test_recover(MATRIX_T, 1U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs100) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(uni_idx(expression(int_literal(5))));

  three_index_errs(idxs);
  three_index_recover(idxs, 2U);
  test_recover(DOUBLE_T, 1U, VECTOR_T, 2U, idxs);
  test_recover(DOUBLE_T, 1U, ROW_VECTOR_T, 2U, idxs);
  test_recover(DOUBLE_T, 1U, MATRIX_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 1U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs101) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(VECTOR_T, 1U, VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 1U, ROW_VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 1U, MATRIX_T, 1U, idxs);
  test_recover(MATRIX_T, 1U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs110) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  three_index_errs(idxs);
  three_index_recover(idxs, 1U);
  test_recover(DOUBLE_T, 2U, VECTOR_T, 2U, idxs);
  test_recover(DOUBLE_T, 2U, ROW_VECTOR_T, 2U, idxs);
  test_recover(VECTOR_T, 1U, MATRIX_T, 1U, idxs);
  test_recover(ROW_VECTOR_T, 2U, MATRIX_T, 2U, idxs);
}
TEST(langAst, idxs111) {
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());
  idxs.push_back(omni_idx());

  three_index_errs(idxs);
  three_index_recover(idxs, 0U);
  test_recover(VECTOR_T, 2U, VECTOR_T, 2U, idxs);
  test_recover(ROW_VECTOR_T, 2U, ROW_VECTOR_T, 2U, idxs);
  test_recover(MATRIX_T, 1U, MATRIX_T, 1U, idxs);
  test_recover(MATRIX_T, 2U, MATRIX_T, 2U, idxs);
}

TEST(langAst, indexOpSliced) {
  using stan::lang::index_op_sliced;
  
  vector<idx> idxs;
  idxs.push_back(omni_idx());
  idxs.push_back(uni_idx(expression(int_literal(3))));

  // no need to retest all of type inference here --- just that it's plumbed
  index_op_sliced ios;
  stan::lang::variable v("foo");
  v.set_type(VECTOR_T, 1U);
  ios.expr_ = v;
  ios.idxs_ = idxs;
  ios.infer_type();
  EXPECT_EQ(DOUBLE_T, ios.type_.base_type_);
  EXPECT_EQ(1U, ios.type_.num_dims_);
}

TEST(langAst, lhsVarOccursOnRhs) {
  stan::lang::variable v("foo");
  v.set_type(DOUBLE_T, 0);
  std::vector<stan::lang::idx> is;
  stan::lang::expression e_int3(stan::lang::int_literal(3));
  stan::lang::uni_idx ui(e_int3);
  stan::lang::idx idx0(ui);
  is.push_back(idx0);
  stan::lang::expression e(stan::lang::int_literal(3));
  stan::lang::assgn a(v, is, e);
  EXPECT_FALSE(a.lhs_var_occurs_on_rhs());

  std::vector<stan::lang::idx> is2;
  stan::lang::assgn a2(v, is2, v);
  EXPECT_TRUE(a2.lhs_var_occurs_on_rhs());

  stan::lang::unary_op uo('+', v);
  stan::lang::assgn a3(v, is2, uo);
  EXPECT_TRUE(a3.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo(v, "-", e_int3);
  stan::lang::assgn a4(v, is2, bo);
  EXPECT_TRUE(a4.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo2(e_int3, "*", e_int3);
  stan::lang::assgn a5(v, is2, bo2);
  EXPECT_FALSE(a5.lhs_var_occurs_on_rhs());

  stan::lang::binary_op bo3(e_int3, "*", bo);
  stan::lang::assgn a6(v, is2, bo3);
  EXPECT_TRUE(a6.lhs_var_occurs_on_rhs());

  stan::lang::index_op_sliced ios(v, is2);
  stan::lang::assgn a7(v, is2, ios);
  EXPECT_TRUE(a7.lhs_var_occurs_on_rhs());
}












