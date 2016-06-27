#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, if_elseDeprecationFunction) {
  test_warning("function-signatures/math/functions/if_else",
               "Warning (non-fatal): the if_else() function is deprecated.");
}
