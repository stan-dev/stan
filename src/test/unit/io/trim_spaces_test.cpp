#include <gtest/gtest.h>
#include <stan/io/trim_spaces.hpp>

TEST(ioUtil, trimSpaces) {
  using stan::io::trim_spaces;
  EXPECT_EQ("", trim_spaces(""));
  EXPECT_EQ("", trim_spaces(" "));
  EXPECT_EQ("", trim_spaces("  "));

  EXPECT_EQ("", trim_spaces("\t"));
  EXPECT_EQ("", trim_spaces("\t\n\r"));

  EXPECT_EQ("a", trim_spaces("a"));
  EXPECT_EQ("a", trim_spaces(" a"));
  EXPECT_EQ("a", trim_spaces("a "));
  EXPECT_EQ("abcd efg", trim_spaces("\t\n      abcd efg     \r\n     "));

}
