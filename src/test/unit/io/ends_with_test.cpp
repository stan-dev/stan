#include <stan/io/ends_with.hpp>
#include <gtest/gtest.h>
#include <string>

void expect_ends(const std::string& p, const std::string& s, bool cond) {
  EXPECT_EQ(cond, stan::io::ends_with(p, s));
}

TEST(Io, EndsWith) {
  expect_ends("", "", true);
  expect_ends("", "a", true);
  expect_ends("a", "a", true);
  expect_ends("a", "    a", true);
  expect_ends("abc", "abc", true);
  expect_ends("bcd", "abcd", true);

  expect_ends("a", "", false);
  expect_ends("a", "ab", false);
  expect_ends("abc", "ab", false);
}
