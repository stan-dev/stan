#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>
#include <exception>

TEST(lang_parser, break_continue_parsable) {
  test_parsable("break-continue");
}

void test_bad_break(const std::string& model_name) {
  test_throws(model_name,
              "Break and continue statements are only allowed"
              " in the body of a for-loop or while-loop.");
}

TEST(lang_parser, break_continue_bad_break1) {
  test_bad_break("break1");
  test_bad_break("continue1");
  test_bad_break("break2");
  test_bad_break("continue2");
  test_bad_break("break3");
  test_bad_break("continue3");
  test_bad_break("break4");
  test_bad_break("continue4");
  test_bad_break("break5");
  test_bad_break("continue5");
  test_bad_break("break6");
  test_bad_break("break7");
  test_bad_break("break8");
}

