#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <stdexcept>

TEST(VariableMap, getFail) {
  stan::lang::variable_map vm;
  EXPECT_THROW(vm.get("x"), std::invalid_argument);
}

TEST(VariableMap, add_get) {
  stan::lang::variable_map vm;
  stan::lang::var_decl x("x", stan::lang::double_type());
  stan::lang::scope x_origin = stan::lang::parameter_origin;
  vm.add("x", x, x_origin);
  stan::lang::var_decl x2 = vm.get("x");
  EXPECT_EQ(x.name_, x2.name_);
  EXPECT_TRUE(x.bare_type_ == x2.bare_type_);
  stan::lang::scope x2_origin = vm.get_scope("x");
  EXPECT_EQ(x_origin.program_block(), x2_origin.program_block());
}

TEST(VariableMap, overwrite) {
  stan::lang::variable_map vm;
  stan::lang::var_decl x("x", stan::lang::double_type());
  stan::lang::scope x_origin = stan::lang::parameter_origin;
  vm.add("x", x, x_origin);
  stan::lang::scope y_origin = stan::lang::data_origin;
  vm.add("x", x, y_origin);
  stan::lang::var_decl x2 = vm.get("x");
  EXPECT_EQ(x.name_, x2.name_);
  EXPECT_TRUE(x.bare_type_ == x2.bare_type_);
  stan::lang::scope x2_origin = vm.get_scope("x");
  EXPECT_EQ(y_origin.program_block(), x2_origin.program_block());
}
