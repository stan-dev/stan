#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, functions_standalone_parsable) {
  test_parsable_standalone_functions("basic");
  test_parsable_standalone_functions("special_functions");
}

// TODO(martincerny) test forward function declarations

// TODO(martincerny) check that the -namespace argument to stanc 
// is used and enforced correctly


