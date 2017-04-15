#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, functions_standalone_parsable) {
  test_parsable_standalone_functions("basic");
}

// TODO(martincerny) check that the -namespace argument is used and enforced
// correctly


