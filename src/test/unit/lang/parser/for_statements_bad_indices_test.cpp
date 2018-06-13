#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>
#include <exception>

TEST(lang_parser, for_statements_bad_indices_test) {
  test_throws("for_statements_bad_indices0", "");
  test_throws("for_statements_bad_indices1", "");
  test_throws("for_statements_bad_indices2", "");
  test_throws("for_statements_bad_indices3", "");
  test_throws("for_statements_bad_indices4", "");
}

