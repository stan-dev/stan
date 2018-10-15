#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <set>
#include <vector>

// test functions needed by stan/lang/ast/sigs/function_signatures
// was part of test/unit/lang/ast_test.hpp

using stan::lang::function_signatures;
using stan::lang::function_signature_t;
using stan::lang::bare_expr_type;
using stan::lang::expression;
using stan::lang::is_user_defined;
using stan::lang::int_literal;
using stan::lang::int_type;
using stan::lang::double_literal;
using stan::lang::double_type;
using std::vector;
using std::string;
using std::pair;
using std::set;


TEST(langAst, isUserDefined) {

  vector<expression> args;
  string name = "foo";
  EXPECT_FALSE(is_user_defined(name, args));
  args.push_back(expression(int_literal(0)));
  EXPECT_FALSE(is_user_defined(name, args));

  vector<bare_expr_type> arg_types;
  arg_types.push_back(bare_expr_type(bare_expr_type(int_type())));
  double_type dt;
  bare_expr_type result_type(dt);
  
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
  function_signatures::reset_sigs();
  // get
  function_signatures& fs1 = function_signatures::instance();
  set<string> ks1 = fs1.key_set();
  size_t keyset_size = ks1.size();
  EXPECT_TRUE(keyset_size > 0);
  // destroy
  function_signatures::reset_sigs();
  // get again
  function_signatures& fs2 = function_signatures::instance();
  set<string> ks2 = fs2.key_set();
  EXPECT_EQ(keyset_size, ks2.size());
}
