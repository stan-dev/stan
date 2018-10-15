#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, if_elseDeprecationFunction) {
  test_warning("function-signatures/math/functions/if_else",
               "Info: the if_else() function is deprecated.");
}
