#include <stan/version.hpp>
#include <stan/gm/compiler.hpp>

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stan/io/cmd_line.hpp>

void print_version(std::ostream* out_stream) {
  if (!out_stream) return;
  *out_stream << "stanc version "
              << stan::MAJOR_VERSION
              << "."
              << stan::MINOR_VERSION
              << "."
              << stan::PATCH_VERSION
              << std::endl;
}

/** 
 * Prints the Stan compiler (stanc) help.
 */
void print_stanc_help(std::ostream* out_stream) {
  using stan::io::print_help_option;
  
  if (!out_stream) return;

  *out_stream << std::endl;
  print_version(out_stream);
  *out_stream << std::endl;

  *out_stream << "USAGE:  " << "stanc [options] <model_file>" << std::endl;
  *out_stream << std::endl;
  
  *out_stream << "OPTIONS:" << std::endl;
  *out_stream << std::endl;
  
  print_help_option(out_stream,"help","","Display this information");

  print_help_option(out_stream,"version","","Display stanc version number");

  print_help_option(out_stream,"name","string",
                    "Model name",
                    "default = \"$model_filename_model\"");

  print_help_option(out_stream,"o","file",
                    "Output file for generated C++ code",
                    "default = \"$name.cpp\"");
  
}

void delete_file(std::ostream* err_stream,
                 const std::string& file_name) {
  int deleted = std::remove(file_name.c_str());
  if (deleted != 0 && file_name.size() > 0)
    if (err_stream) 
      std::cerr << "Could not remove output file=" << file_name
                << std::endl;
}      


int stanc_helper(int argc, const char* argv[], 
                 std::ostream* out_stream, std::ostream* err_stream) {
  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;
  static const int INVALID_ARGUMENT_RC = -3;

  std::string out_file_name; // declare outside of try to delete in catch

  try {

    stan::io::cmd_line cmd(argc,argv);

    if (cmd.has_flag("help")) {
      print_stanc_help(out_stream);
      return SUCCESS_RC;
    }

    if (cmd.has_flag("version")) {
      print_version(out_stream);
      return SUCCESS_RC;
    }

    if (cmd.bare_size() != 1) {
      std::string msg("Require model file as argument. ");
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
    std::fstream out(out_file_name.c_str(),
                     std::fstream::out);
    if (out_stream) {
      *out_stream << "Model name=" << model_name << std::endl;
      *out_stream << "Input file=" << in_file_name << std::endl;
      *out_stream << "Output file=" << out_file_name << std::endl;
    }
    bool include_main = !cmd.has_flag("no_main");
    bool valid_model 
      = stan::gm::compile(err_stream,in,out,model_name,include_main,in_file_name);
    out.close();
    if (!valid_model) {
      if (err_stream)
        *err_stream << "PARSING FAILED." << std::endl;
      delete_file(out_stream,out_file_name);  // FIXME: how to remove triple cut-and-paste?
      return PARSE_FAIL_RC;
    }
  } catch (const std::invalid_argument& e) {
    if (err_stream) {
      *err_stream << std::endl
                  << e.what()
                  << std::endl;
      *err_stream << "Execute \"stanc --help\" for more information" 
                  << std::endl;
      delete_file(out_stream,out_file_name);
    }
    return INVALID_ARGUMENT_RC;
  } catch (const std::exception& e) {
    if (err_stream) {
      *err_stream << std::endl
                  << e.what()
                  << std::endl;
    }
    delete_file(out_stream,out_file_name);
    return EXCEPTION_RC;
  }
  
  return SUCCESS_RC;
}

