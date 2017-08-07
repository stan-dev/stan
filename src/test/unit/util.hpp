#ifndef TEST_UNIT_UTIL_HPP
#define TEST_UNIT_UTIL_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>

#define EXPECT_THROW_MSG(expr, T_e, msg)                \
  try {                                                 \
    expr;                                               \
  } catch(const T_e& e) {                               \
    EXPECT_EQ(1, count_matches(msg, e.what()))          \
      << "expected message: " << msg << std::endl       \
      << "found message:    " << e.what();              \
  } catch(...) {                                        \
      FAIL()                                            \
        << "Wrong exception type thrown" << std::endl;  \
  }


int count_matches(const std::string& target,
                  const std::string& s) {
  if (target.size() == 0) return -1;  // error
  int count = 0;
  for (size_t pos = 0; (pos = s.find(target,pos)) != std::string::npos; pos += target.size())
    ++count;
  return count;
}

namespace stan {
  namespace test {

    std::streambuf *cout_buf = 0;
    std::streambuf *cerr_buf = 0;

    std::stringstream cout_ss;
    std::stringstream cerr_ss;

    void capture_std_streams() {
      cout_ss.str("");
      cerr_ss.str("");

      cout_buf = std::cout.rdbuf();
      cerr_buf = std::cerr.rdbuf();

      std::cout.rdbuf(cout_ss.rdbuf());
      std::cerr.rdbuf(cerr_ss.rdbuf());
    }

    void reset_std_streams() {
      std::cout.rdbuf(cout_buf);
      std::cerr.rdbuf(cerr_buf);
      cout_buf = 0;
      cerr_buf = 0;
    }

    /**
     * Compare the two arguments using EXPECT_EQ.
     */
    template <typename T1, typename T2>
    void expect_eq(const T1& x1, const T2& x2) {
      EXPECT_EQ(x1, x2);
    }

    /**
     * Compare the two containers for size using EXPECT_EQ and then
     * recursively compare the elements using expect_eq.
     */
    template <typename T>
    void expect_eq(const std::vector<T>& x, const std::vector<T>& y) {
      EXPECT_EQ(x.size(), y.size());
      for (size_t i = 0; i < x.size(); ++i)
        expect_eq(x[i], y[i]);
    }

  }
}
#endif
