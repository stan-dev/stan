#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, algebra_solver_function_signatures) {
  test_parsable("function-signatures/math/matrix/algebra_solver");
}
