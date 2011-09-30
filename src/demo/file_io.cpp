#include <fstream>
#include <iostream>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>

int main(int argc, const char* argv[]) {
  stan::io::cmd_line cmd(argc,argv);
  std::ifstream f("src/models/norm_sample.dmp");
  if (!f.is_open()) {
    std::cout << "COULD NOT OPEN FILE" << std::endl;
    return -1;
  }
  stan::io::dump d(f);
}
