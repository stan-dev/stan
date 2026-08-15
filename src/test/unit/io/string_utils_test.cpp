#include <gtest/gtest.h>
#include <stan/io/string_utils.hpp>
#include <string>
#include <string_view>
#include <vector>

using stan::io::contains;
using stan::io::join;
using stan::io::replace_first;
using stan::io::split;
using stan::io::starts_with;
using stan::io::trim;

TEST(io_string_utils, split_simple) {
  EXPECT_EQ((std::vector<std::string>{"a", "b"}), split("a.b", "."));
  EXPECT_EQ((std::vector<std::string>{"abc"}), split("abc", "."));
  EXPECT_EQ((std::vector<std::string>{""}), split("", "."));
  EXPECT_EQ((std::vector<std::string>{"a", "", "b"}), split("a,,b", ","));
  EXPECT_EQ((std::vector<std::string>{"", "a", "b", ""}), split(",a,b,", ","));
}

TEST(io_string_utils, split_multi_char_delims) {
  EXPECT_EQ((std::vector<std::string>{"a", "b", "c"}), split("a,b;c", ",;"));
}

TEST(io_string_utils, split_accepts_views) {
  std::string_view sv("x.y.z");
  EXPECT_EQ((std::vector<std::string>{"x", "y", "z"}), split(sv, "."));
  std::string str("x.y");
  EXPECT_EQ((std::vector<std::string>{"x", "y"}), split(str, "."));
  EXPECT_EQ((std::vector<std::string>{"x", "y"}),
            split(std::string_view(str), "."));
}

TEST(io_string_utils, split_compress) {
  EXPECT_EQ((std::vector<std::string>{"a", "b"}), split("a,,b", ",", true));
  EXPECT_EQ((std::vector<std::string>{"a", "b"}), split("a,,,b", ",", true));
  EXPECT_EQ((std::vector<std::string>{"", "a", "b", ""}),
            split(",a,,b,", ",", true));
  EXPECT_EQ((std::vector<std::string>{"", ""}), split(",,", ",", true));
  EXPECT_EQ((std::vector<std::string>{""}), split("", ",", true));
  EXPECT_EQ((std::vector<std::string>{"a"}), split("a", ",", true));
}

TEST(io_string_utils, join) {
  EXPECT_EQ("", join({}, "."));
  EXPECT_EQ("a", join({"a"}, "."));
  EXPECT_EQ("a.b.c", join({"a", "b", "c"}, "."));
  EXPECT_EQ("a,,b", join({"a", "", "b"}, ","));
  EXPECT_EQ("xy", join({"x", "y"}, ""));
}

TEST(io_string_utils, trim) {
  std::string s = "  abc \t\n ";
  trim(s);
  EXPECT_EQ("abc", s);
  s = "\t \n";
  trim(s);
  EXPECT_EQ("", s);
  s = "abc";
  trim(s);
  EXPECT_EQ("abc", s);
}

TEST(io_string_utils, replace_first) {
  std::string s = "hello (Default) world (Default)";
  replace_first(s, " (Default)", "");
  EXPECT_EQ("hello world (Default)", s);
  replace_first(s, "missing", "x");
  EXPECT_EQ("hello world (Default)", s);
  replace_first(s, "world", "stan");
  EXPECT_EQ("hello stan (Default)", s);
  replace_first(s, "hello", "");
  EXPECT_EQ(" stan (Default)", s);
}

TEST(io_string_utils, contains) {
  EXPECT_TRUE(contains("hello world", "lo wo"));
  EXPECT_TRUE(contains("hello world", ""));
  EXPECT_TRUE(contains("hello world", "hello world"));
  EXPECT_FALSE(contains("hello world", "missing"));
  EXPECT_FALSE(contains("", "x"));
}

TEST(io_string_utils, starts_with) {
  EXPECT_TRUE(starts_with("hello world", "hello"));
  EXPECT_TRUE(starts_with("hello world", ""));
  EXPECT_TRUE(starts_with("hello world", "hello world"));
  EXPECT_FALSE(starts_with("hello world", "hello world!"));
  EXPECT_FALSE(starts_with("hello world", "Hello"));
  EXPECT_FALSE(starts_with("", "x"));
}
