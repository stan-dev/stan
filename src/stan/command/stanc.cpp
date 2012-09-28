#include <stan/version.hpp>
#include <stan/gm/compiler.hpp>

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stan/io/cmd_line.hpp>

/** 
 * Prints the Stan version to std::cout.
 * 
 */
void print_version() {
  std::cout << "stanc version "
            << stan::MAJOR_VERSION
            << "."
            << stan::MINOR_VERSION
            << "."
            << stan::PATCH_VERSION
            << std::endl;
}

/** 
 * Prints the Stan compiler (stanc) help 
 * to std::cout.
 * 
 */
void print_stanc_help() {
  using stan::io::print_help_option;

  std::cout << std::endl;
  print_version();
  std::cout << std::endl;

  std::cout << "USAGE:  " << "stanc [options] <model_file>" << std::endl;
  std::cout << std::endl;
  
  std::cout << "OPTIONS:" << std::endl;
  std::cout << std::endl;
  
  print_help_option("help","","Display this information");

  print_help_option("version","","Display stanc version number");

  print_help_option("name","string",
                    "Model name",
                    "default = \"$model_filename_model\"");

  print_help_option("o","file",
                    "Output file for generated C++ code",
                    "default = \"$name.cpp\"");
  
}

void delete_file(const std::string& file_name) {
  int deleted = std::remove(file_name.c_str());
  std::cout << (deleted == 0
                ? "OUTPUT FILE REMOVED"
                : "COULD NOT REMOVE OUTPUT FILE")
            << ", file=" << file_name.c_str() 
            << std::endl;
}      


/** 
 * The Stan compiler (stanc).
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * 
 * @return 0 for success, -1 for exception, -2 for parser failure
 */
int main(int argc, const char* argv[]) {

  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;

  stan::io::cmd_line cmd(argc,argv);

  if (cmd.has_flag("help")) {
    print_stanc_help();
    return SUCCESS_RC;
  }

  if (cmd.has_flag("version")) {
    print_version();
    return SUCCESS_RC;
  }

  if (cmd.bare_size() != 1) {
    std::string msg("require file name to compile as input. "
                    "execute \"stanc --help\" for more information");
    throw std::invalid_argument(msg);
  }
  std::string in_file_name;
  cmd.bare(0,in_file_name);
  std::fstream in(in_file_name.c_str());

  std::string model_name;
  if (cmd.has_key("name")) {
    cmd.val("name",model_name);
  } else {
    size_t slashInd = in_file_name.rfind('/');
    size_t ptInd = in_file_name.rfind('.');
    if (ptInd == std::string::npos)
      ptInd = in_file_name.length();
    if (slashInd == std::string::npos) {
      slashInd = in_file_name.rfind('\\');
    }
    if (slashInd == std::string::npos) {
      slashInd = 0;
    } else {
      slashInd++;
    }
    model_name = in_file_name.substr(slashInd,ptInd - slashInd) + "_model";
    for (std::string::iterator strIt = model_name.begin();
         strIt != model_name.end(); strIt++) {
      if (!isalnum(*strIt) && *strIt != '_') {
        *strIt = '_';
      }
    }
  }

  std::string out_file_name;
  if (cmd.has_key("o")) {
    cmd.val("o",out_file_name);
  } else {
    out_file_name = model_name;
    out_file_name += ".cpp";
  }

  if (!isalpha(model_name[0]) && model_name[0] != '_') {
    std::string msg("model_name must not start with a number or symbol other than _");
    throw std::invalid_argument(msg);
  }
  for (std::string::iterator strIt = model_name.begin();
       strIt != model_name.end(); strIt++) {
    if (!isalnum(*strIt) && *strIt != '_') {
      std::string msg("model_name must contain only letters, numbers and _");
      throw std::invalid_argument(msg);
    }
  }
    
  stan::gm::program prog;
  try {
    std::fstream out(out_file_name.c_str(),
                     std::fstream::out);
    std::cout << "Model name=" << model_name << std::endl;
    std::cout << "Input file=" << in_file_name << std::endl;
    std::cout << "Output file=" << out_file_name << std::endl;
    bool include_main = !cmd.has_flag("no_main");
    bool valid_model 
      = stan::gm::compile(in,out,model_name,include_main,in_file_name);
    out.close();
    if (!valid_model) {
      std::cout << "PARSING FAILED." << std::endl;
      delete_file(out_file_name);
      return PARSE_FAIL_RC;
    }
  } catch (const std::exception& e) {
    std::cout << std::endl
              << "ERROR PARSING"
              << std::endl
              << e.what()
              << std::endl;
    delete_file(out_file_name);
    return EXCEPTION_RC;
  }
  
  return SUCCESS_RC;
}

