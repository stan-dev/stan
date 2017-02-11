#include <stan/io/program_reader.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>


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

  vector<string> search_path;
  search_path.push_back("/Users/carp/foo/");
  search_path.push_back("/Users/carp/temp2/");
  search_path.push_back("/Users/carp/bar/");

  stan::io::program_reader reader(ss, "str_test0", search_path);

  std::cout << std::endl << "SURVEY SAYS:" << std::endl;
  std::cout << reader.program_stream().rdbuf();
  std::cout << std::endl << "HISTORY:" << std::endl;
  reader.print_history(std::cout);

  stan::io::program_reader::dumps_t evs = reader.include_stack(3);
  std::string dump = stan::io::program_reader::render(evs);
  std::cout << std::endl << "INCLUDE DUMP:" << std::endl << dump << std::endl;
}


TEST(prog_reader, one) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include foo.stan\n"
     << "}\n"
     << "model {\n"
     << "}\n"
     << "";

  vector<string> search_path;
  search_path.push_back("/Users/carp/foo/");
  search_path.push_back("/Users/carp/temp2/");
  search_path.push_back("/Users/carp/bar/");

  stan::io::program_reader reader(ss, "my_path_is_a_string", search_path);

  std::cout << std::endl << "SURVEY SAYS:" << std::endl;
  std::cout << reader.program_stream().rdbuf();
  std::cout << std::endl << "HISTORY:" << std::endl;
  reader.print_history(std::cout);

  stan::io::program_reader::dumps_t evs = reader.include_stack(3);
  std::string dump = stan::io::program_reader::render(evs);
  std::cout << std::endl << "INCLUDE DUMP:" << std::endl << dump << std::endl;

}
