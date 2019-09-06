#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <boost/variant/polymorphic_get.hpp>
#include <cmath>

#include <string>

TEST(StanLangAst, MapRect) {
  using stan::lang::int_literal;

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
