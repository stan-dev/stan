#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, illegalScope) { 
  test_throws("get_lp_bad_scope1",
              "lp suffixed functions only allowed in ");
  test_throws("get_lp_bad_scope2",
              "lp suffixed functions only allowed in ");
}
TEST(langParser, legal) {
  test_parsable("get_lp_good");
}
