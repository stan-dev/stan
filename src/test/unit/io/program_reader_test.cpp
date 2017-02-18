#include <stan/io/program_reader.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

std::vector<std::string> create_search_path() {
  std::vector<std::string> search_path;
  search_path.push_back("foo");
  search_path.push_back("src/test/test-models/included/");
  search_path.push_back("bar/baz");
  return search_path;
}

TEST(prog_reader, zero) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "parameters {\n"
     << "  real y;\n"
     << "}\n"
     << "model {\n"
     << "  y ~ normal(0, 1);\n"
     << "}\n"
     << "";

  vector<string> search_path = create_search_path();

  stan::io::program_reader reader(ss, "str_test0", search_path);

  std::cout << std::endl << "SURVEY SAYS:" << std::endl;
  std::cout << reader.program_stream().rdbuf();
  std::cout << std::endl << "HISTORY:" << std::endl;
  reader.print_history(std::cout);

  std::string dump = reader.include_trace(4);
  std::cout << std::endl << "INCLUDE DUMP:" << std::endl << dump << std::endl;
}


TEST(prog_reader, one) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include incl_fun.stan\n"
     << "}\n"
     << "#include incl_params.stan\n"
     << "model {\n"
     << "}\n";

  vector<string> search_path = create_search_path();

  stan::io::program_reader reader(ss, "my_path_is_a_string", search_path);

  std::cout << std::endl << "SURVEY SAYS:" << std::endl;
  std::cout << reader.program_stream().rdbuf();
  std::cout << std::endl << "HISTORY:" << std::endl;
  reader.print_history(std::cout);

  for (int i = 1; i <= 10; ++i) {
    std::string dump = reader.include_trace(i);
    std::cout << "INCLUDE DUMP(" << i << ") : " << std::endl
              << dump << std::endl;
  }

/*

 1 functions {            // top at 1
 2  int foo() {           // incl at 1, from top at 2
 3    return 1;           // incl at 2, from top at 2
 4  }                     // incl at 3, from top at 2
 5 }                      // top at 3
 6 parameters {           // params at 1, from top at 4
 7   real y;              // params at 2, from top at 4
 8 }                      // params at 3, from top at 4
 9 model {                // top at 4
10 }                      // top at 5

*/

}
