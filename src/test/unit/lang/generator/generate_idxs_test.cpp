#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

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
  using stan::lang::bare_expr_type;
  using stan::lang::double_type;

  double_type tDouble;
  bare_expr_type betDouble(tDouble);
  stan::lang::variable v("foo");
  v.set_type(betDouble);

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
  stan::lang::assgn a(v, is, "=", e);
  stan::lang::statement s(a);
  s.begin_line_ = 12U;
  s.end_line_ = 14U;

  std::stringstream o;
  generate_statement(s, 2, o);
  EXPECT_TRUE(0U < o.str().find(
      "stan::model::cons_list(stan::model::index_uni(3), stan::model::cons_list(stan::model::index_max(5), stan::model::nil_index_list()))"));
}
