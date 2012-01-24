#include "stan/gm/compiler.hpp"

#include <string>
#include <exception>

int main() {
  static int SUCCESS_RC = 0;
  static int EXCEPTION_RC = -1;
  static int PARSE_FAIL_RC = -2;

  stan::gm::program prog;
  try {
    bool ok
      = stan::gm::compile(std::cin,std::cout,"demo_model");

    if (!ok) {
      std::cout << "PARSING FAILED." << std::endl;
      return PARSE_FAIL_RC;
    }
  } catch(const std::exception& e) {
    std::cerr << std::endl
              << "ERROR PARSING"
              << std::endl
              << e.what();
    return EXCEPTION_RC;
  }
  
  return SUCCESS_RC;
}

