#ifndef TEST__UNIT__UTIL_HPP
#define TEST__UNIT__UTIL_HPP

#include <string>

#define EXPECT_THROW_MSG(expr, T_e, msg)                \
  EXPECT_THROW(expr, T_e);                              \
  try {                                                 \
    expr;                                               \
  } catch(const T_e& e) {                               \
    EXPECT_EQ(1, count_matches(msg, e.what()))          \
      << "expected message: " << msg << std::endl       \
      << "found message:    " << e.what();              \
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
  }
}

#endif
