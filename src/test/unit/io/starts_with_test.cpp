#include <stan/io/starts_with.hpp>
#include <gtest/gtest.h>
#include <string>

void expect_starts(const std::string& p, const std::string& s, bool cond) {
  EXPECT_EQ(cond, stan::io::starts_with(p, s));
}

TEST(Io, StartsWith) {
  expect_starts("", "", true);
  expect_starts("", "a", true);
  expect_starts("abc", "abc", true);
  expect_starts("abc", "abcd", true);

  expect_starts("a", "", false);
  expect_starts("abc", "ab", false);
}
