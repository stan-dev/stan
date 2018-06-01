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
  test_throws("for_loops/for_statements_bad_indices0", "");
  test_throws("for_loops/for_statements_bad_indices1", "");
  test_throws("for_loops/for_statements_bad_indices2", "");
  test_throws("for_loops/for_statements_bad_indices3", "");
  test_throws("for_loops/for_statements_bad_indices4", "");
}

TEST(langParserForLoop, BadLoopVarAssign) {
  test_throws("for_loops/assign_to_loop_var1", "");
  test_throws("for_loops/assign_to_loop_var2", "");
  test_throws("for_loops/assign_to_loop_var3", "");
  test_throws("for_loops/assign_to_loop_var4", "");
  test_throws("for_loops/assign_to_loop_var5", "");
}

