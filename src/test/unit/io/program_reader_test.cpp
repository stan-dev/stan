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
  std::stringstream ss;
  ss << "parameters {\n"            // 1
     << "  real y;\n"               // 2
     << "}\n"                       // 3
     << "model {\n"                 // 4
     << "  y ~ normal(0, 1);\n"     // 5
     << "}\n"                       // 6
     << "";                         // 7 (nothing on line)

  std::vector<std::string> search_path = create_search_path();
  stan::io::program_reader reader(ss, "foo", search_path);
  EXPECT_EQ("in file 'foo' at line 1\n", reader.include_trace(1));
  EXPECT_EQ("in file 'foo' at line 2\n", reader.include_trace(2));
  EXPECT_EQ("in file 'foo' at line 3\n", reader.include_trace(3));
  EXPECT_EQ("in file 'foo' at line 4\n", reader.include_trace(4));
  EXPECT_EQ("in file 'foo' at line 5\n", reader.include_trace(5));
  EXPECT_EQ("in file 'foo' at line 6\n", reader.include_trace(6));
  EXPECT_THROW(reader.include_trace(7), std::runtime_error);
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
}


TEST(prog_reader, two) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include incl_rec.stan\n"
     << "}\n"
     << "model { }\n";

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
}

