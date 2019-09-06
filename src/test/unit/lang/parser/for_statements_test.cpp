#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>
#include <exception>

TEST(langParserForLoop, ContainerLoop1) {
  test_parsable("for_loops/x_in_xs_function_block_1");
}

TEST(langParserForLoop, ContainerLoop2) {
  test_parsable("for_loops/x_in_xs_function_block_2");
}

TEST(langParserForLoop, ContainerLoop3) {
  test_parsable("for_loops/x_in_xs_function_block_3");
}

TEST(langParserForLoop, BadIndices) {
  test_throws("for_loops/for_statements_bad_indices0",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/for_statements_bad_indices1",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/for_statements_bad_indices2",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/for_statements_bad_indices3",
  "Loop must be over container or range.");
  test_throws("for_loops/for_statements_bad_indices4",
  "Loop variable already declared.");
}

TEST(langParserForLoop, BadLoopVarAssign) {
  test_throws("for_loops/assign_to_loop_var1",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var2",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var3",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var4",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var5",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var6",
  "Loop variable v cannot be used on left side of assignment statement");
  test_throws("for_loops/assign_to_loop_var7",
  "Loop variable v cannot be used on left side of assignment statement");
}

