#include <fstream>
#include <iostream>
#include <gtest/gtest.h>

void read_file(const std::string& path,
               std::string& contents) {
  std::stringstream s;
  std::fstream f(path.c_str());
  while (f.good())
    s << static_cast<char>(f.get());
  contents = s.str();
}

void test_pg(const std::string& program_name,
             const std::string& expected_substring) {
  std::string path = "src/test/test-models/good/parser-generator";
  path += "/";
  path += program_name;
  path += ".cpp";

  std::string cpp_code;
  read_file(path, cpp_code);
  // std::cout << "cpp_code=" << cpp_code << std::endl;
  
  EXPECT_TRUE(cpp_code.find(expected_substring) != std::string::npos)
    << "program_name: " << program_name << std::endl
    << "expected_substring: " << expected_substring;
}

int count_occurrences(const std::string target,
                      const std::string s) {
  int count = 0;
  int offset = 0;
  while (true) {
    offset = s.find(target, offset);
    if (offset == std::string::npos) break;
    offset += target.size();
    ++count;
  }
  return count;
}

void test_pg_count(const std::string& program_name,
                   const std::string& expected_substring,
                   const int expected_count) {
  std::string path = "src/test/test-models/good/parser-generator";
  path += "/";
  path += program_name;
  path += ".cpp";

  std::string cpp_code;
  read_file(path, cpp_code);
  // std::cout << "cpp_code=" << cpp_code << std::endl;
  

  EXPECT_EQ(expected_count, count_occurrences(expected_substring,cpp_code));
}

TEST(unitGm, simpleTest) {
  test_pg("user-function-struct-const", 
          "operator()(const T0__& x, std::ostream* pstream__) const {");
}     
             
TEST(unitGm, odeTest) {
  std::string expected;
  expected = "stan::math::assign(y_hat, "
    "integrate_ode(sho_functor__(), y0, t0, ts, theta, x, x_int, pstream__));";
  test_pg("ode", expected);
  test_pg_count("ode", expected, 2);
}
             
