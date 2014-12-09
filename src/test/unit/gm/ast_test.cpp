#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <set>
#include <stan/gm/ast_def.cpp>
#include <gtest/gtest.h>

using stan::gm::function_signatures;
using stan::gm::expr_type;
using stan::gm::DOUBLE_T;
using stan::gm::INT_T;
using stan::gm::VECTOR_T;
using stan::gm::ROW_VECTOR_T;
using stan::gm::MATRIX_T;

TEST(gm_ast,expr_type_num_dims) {
  EXPECT_EQ(0U,expr_type().num_dims());
  EXPECT_EQ(2U,expr_type(INT_T,2U).num_dims());
  EXPECT_EQ(2U,expr_type(VECTOR_T,2U).num_dims());
}
TEST(gm_ast,expr_type_is_primitive) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive());
  EXPECT_TRUE(expr_type(INT_T).is_primitive());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive());
}
TEST(gm_ast,expr_type_is_primitive_int) {
  EXPECT_FALSE(expr_type(DOUBLE_T).is_primitive_int());
  EXPECT_TRUE(expr_type(INT_T).is_primitive_int());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_int());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_int());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_int());
}
TEST(gm_ast,expr_type_is_primitive_double) {
  EXPECT_TRUE(expr_type(DOUBLE_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T).is_primitive_double());
  EXPECT_FALSE(expr_type(VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(ROW_VECTOR_T).is_primitive_double());
  EXPECT_FALSE(expr_type(MATRIX_T).is_primitive_double());
  EXPECT_FALSE(expr_type(INT_T,2U).is_primitive_double());
}
TEST(gm_ast,expr_type_eq) {
  EXPECT_EQ(expr_type(DOUBLE_T),expr_type(DOUBLE_T));
  EXPECT_EQ(expr_type(DOUBLE_T,1U),expr_type(DOUBLE_T,1U));
  EXPECT_NE(expr_type(INT_T), expr_type(DOUBLE_T));
  EXPECT_NE(expr_type(INT_T,1), expr_type(INT_T,2));
}
TEST(gm_ast,expr_type_type) {
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

TEST(gm_ast,function_signatures_log_sum_exp_1) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T,1U)),
                               error_msgs));
}

TEST(gm_ast,function_signatures_log_sum_exp_2) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
  std::stringstream error_msgs;
  EXPECT_EQ(expr_type(DOUBLE_T),
            fs.get_result_type("log_sum_exp",
                               expr_type_vec(expr_type(DOUBLE_T),
                                             expr_type(DOUBLE_T)),
                               error_msgs));
}

TEST(gm_ast,function_signatures_add) {
  stan::gm::function_signatures& fs = stan::gm::function_signatures::instance();
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

TEST(gmAst,voidType) {
  EXPECT_EQ(stan::gm::VOID_T, 0);
  std::stringstream ss;
  stan::gm::write_base_expr_type(ss, stan::gm::VOID_T);
  EXPECT_EQ("void",ss.str());

  expr_type et(stan::gm::VOID_T,0);
  EXPECT_TRUE(et.is_void());
}

TEST(gmAst,baseVarDecl) {
  std::vector<stan::gm::expression> dims;
  dims.push_back(stan::gm::expression(stan::gm::int_literal(0)));
  stan::gm::base_var_decl bvd("foo", dims, INT_T);
  EXPECT_EQ("foo",bvd.name_);
  EXPECT_EQ(1U, bvd.dims_.size());
  EXPECT_EQ(stan::gm::expression(stan::gm::int_literal(0)).expression_type(),
            bvd.dims_[0].expression_type());
  EXPECT_EQ(INT_T, bvd.base_type_);
}

TEST(gmAst,argDecl) {
  stan::gm::arg_decl ad;
  ad.arg_type_ = expr_type(INT_T,0);
  ad.name_ = "foo";
  stan::gm::base_var_decl bvd = ad.base_variable_declaration();
  EXPECT_EQ("foo", bvd.name_);
  EXPECT_EQ(0U, bvd.dims_.size());
  EXPECT_EQ(INT_T, bvd.base_type_);
}

TEST(gmAst,functionDeclDef) {
  stan::gm::function_decl_def fdd(expr_type(stan::gm::INT_T,0),
                                  "foo",
                                  std::vector<stan::gm::arg_decl>(),
                                  stan::gm::statement(stan::gm::no_op_statement()));
  EXPECT_EQ("foo",fdd.name_);
  EXPECT_TRUE(fdd.body_.is_no_op_statement());
  EXPECT_EQ(0U,fdd.arg_decls_.size());
  EXPECT_TRUE(fdd.return_type_.is_primitive_int());
}
TEST(gmAst,functionDeclDefs) {
  stan::gm::function_decl_def fdd1(expr_type(stan::gm::INT_T,0),
                                  "foo",
                                  std::vector<stan::gm::arg_decl>(),
                                  stan::gm::statement(stan::gm::no_op_statement()));
  stan::gm::arg_decl ad;
  ad.arg_type_ = expr_type(INT_T,0);
  ad.name_ = "foo";
  std::vector<stan::gm::arg_decl> arg_decls;
  arg_decls.push_back(ad);
  stan::gm::function_decl_def fdd2(expr_type(stan::gm::DOUBLE_T,3),
                                  "bar",
                                   arg_decls,
                                   stan::gm::statement(stan::gm::no_op_statement()));
  std::vector<stan::gm::function_decl_def> vec_fdds;
  vec_fdds.push_back(fdd1);
  vec_fdds.push_back(fdd2);
  stan::gm::function_decl_defs fdds(vec_fdds);
  EXPECT_EQ(2U,fdds.decl_defs_.size());
}

TEST(gmAst, hasRngSuffix) {
  EXPECT_TRUE(stan::gm::has_rng_suffix("foo_rng"));
  EXPECT_FALSE(stan::gm::has_rng_suffix("foo.rng"));
  EXPECT_FALSE(stan::gm::has_rng_suffix("foo.bar"));
}
TEST(gmAst, hasLpSuffix) {
  EXPECT_TRUE(stan::gm::has_lp_suffix("foo_lp"));
  EXPECT_FALSE(stan::gm::has_lp_suffix("foo.lp"));
  EXPECT_FALSE(stan::gm::has_lp_suffix("foo.bar"));
}

TEST(gmAst, isUserDefined) {
  using stan::gm::function_signature_t;
  using stan::gm::expr_type;
  using stan::gm::expression;
  using stan::gm::is_user_defined;
  using stan::gm::int_literal;
  using stan::gm::double_literal;
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

TEST(gmAst, resetSigs) {
  using std::set;
  using std::string;

  stan::gm::function_signatures::reset_sigs();

  // test can get, destroy, then get
  stan::gm::function_signatures& fs1
    = stan::gm::function_signatures::instance();
  set<string> ks1 = fs1.key_set();
  int keyset_size = ks1.size();
  EXPECT_TRUE(keyset_size > 0);
  
  stan::gm::function_signatures::reset_sigs();

  stan::gm::function_signatures& fs2
    = stan::gm::function_signatures::instance();

  set<string> ks2 = fs2.key_set();
  EXPECT_EQ(keyset_size,ks2.size());

}

TEST(gmAst, solveOde) {
  using stan::gm::integrate_ode;
  using stan::gm::variable;
  using stan::gm::expr_type;
  using stan::gm::expression;

  integrate_ode so; // null ctor should work and not raise error

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
  integrate_ode so2(system_function_name, y0, t0, ts, theta, x, x_int);

  // dumb test to make sure we at least get the right types back
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
