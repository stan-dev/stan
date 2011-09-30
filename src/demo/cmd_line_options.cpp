#include <iostream>

#include <stan/io/cmd_line.hpp>


int main(int argc, const char* argv[]) {
  stan::io::cmd_line cl(argc, argv);
  cl.print(std::cout);

}
