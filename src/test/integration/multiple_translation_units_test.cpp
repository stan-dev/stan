#include <gtest/gtest.h>

TEST(multiple_translation_units, compile) {
  SUCCEED()
    << "this test compiling indicates that compiling the stan library "
    << "with multiple translation units is ok.";
}
