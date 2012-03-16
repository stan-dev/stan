#include <stan/gm/compiler.hpp>

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stan/io/cmd_line.hpp>

void print_stanc_help() {
  using stan::io::print_help_option;

  std::cout << std::endl;
  std::cout << "Stan Compiler" << std::endl;
  std::cout << std::endl;

  std::cout << "USAGE:  " << "stanc [options] <model_file>" << std::endl;
  std::cout << std::endl;
  
  std::cout << "OPTIONS:" << std::endl;
  std::cout << std::endl;
  
  print_help_option("help","","Display this information");

  print_help_option("name","string",
                    "Model name",
                    "default = \"anon_model\"");

  print_help_option("o","file",
                    "Output file for generated C++ code",
                    "default = \"$name.cpp\"");
  
}

int main(int argc, const char* argv[]) {

  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;

  stan::io::cmd_line cmd(argc,argv);

  if (cmd.has_flag("help")) {
    print_stanc_help();
    return SUCCESS_RC;
  }

  

  stan::gm::program prog;
  try {

    std::string model_name;
    if (cmd.has_key("name")) {
      cmd.val("name",model_name);
    } else {
      model_name = "anon_model";
    }
    
    if (cmd.bare_size() != 1) {
      std::string msg("require file name to compile as input");
      throw std::invalid_argument(msg);
    }
    std::string in_file_name;
    cmd.bare(0,in_file_name);
    std::fstream in(in_file_name.c_str());

    std::string out_file_name = model_name;
    out_file_name += ".cpp";
    std::fstream out(out_file_name.c_str());

    bool valid_model 
      = stan::gm::compile(in,out,model_name);

    if (!valid_model) {
      std::cout << "PARSING FAILED." << std::endl;
      return PARSE_FAIL_RC;
    }

  } catch(const std::exception& e) {
    std::cerr << std::endl
              << "ERROR PARSING"
              << std::endl
              << e.what()
              << std::endl;
    return EXCEPTION_RC;
  }
  
  return SUCCESS_RC;
}

