#include <stan/io/cmd_line.hpp>
#include <stan/gm/compiler.hpp>

#include <iostream>
#include <string>
#include <exception>

/**
 * Compile a Stan graphical model specification to executable
 * code.
 *
 * @return Return code for operation, with 
 *    0: success,
 *   -1: parse exception
 *   -2: unknown error  
 */
int main(int argc, char* argv[]) {

  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;

  stan::gm::program prog;
  try {

    bool ok
      = stan::gm::compile(std::cin,std::cout,"demo_model");

    if (!ok) {
      std::cout << "PARSING FAILED." << std::endl;
      return PARSE_FAIL_RC;
    }
  } catch (const std::exception& e) {
    std::cerr << std::endl
              << "ERROR PARSING"
              << std::endl
              << e.what();
    return EXCEPTION_RC;
  }
  
  return SUCCESS_RC;
}

