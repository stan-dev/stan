#include <stan/lang/located_exception.hpp>
#include <gtest/gtest.h>

TEST(langLocatedException, what) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  // makes sure nested error and line number are included in what() message
  domain_error de("foo");
  located_exception<domain_error> le(de, 3);
  string de_what = de.what();
  string le_what = le.what();
  EXPECT_TRUE(le_what.find_first_of(de_what) != string::npos);
  EXPECT_TRUE(le_what.find_first_of("3") != string::npos);
}

TEST(langLocatedException, catchable) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  // makes sure nested type and located type are catchable
  domain_error de("foo");
  located_exception<domain_error> le(de, 3);
  EXPECT_THROW(throw le, domain_error);
  EXPECT_THROW(throw le, located_exception<domain_error>);
}

TEST(langLocatedException, get) {
 using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  // makes sure get returns reference, not copy
  domain_error de("foo");
  located_exception<domain_error> le(de, 3);
  EXPECT_EQ(&de, &le.nested_exception());
}


TEST(langLocatedException, line) {
 using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  // makes sure get returns reference, not copy
  domain_error de("foo");
  located_exception<domain_error> le(de, 3);
  EXPECT_EQ(3, le.line());
}

TEST(langLocatedException, factory) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;
  using stan::lang::throw_located_exception;

  domain_error de("foo");
  try {
     throw_located_exception(de,17);
  } catch (...) {
    SUCCEED();
  }

}
