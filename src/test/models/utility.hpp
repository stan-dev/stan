#ifndef __TEST__MODELS__UTILITY_HPP__
#define __TEST__MODELS__UTILITY_HPP__

#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

/** 
 * Gets the path separator for the OS.
 * 
 * @return '\' for Windows, '/' otherwise.
 */
char get_path_separator() {
  static char path_separator = 0;
  if (path_separator == 0) {
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    path_separator = fgetc(in);
    pclose(in);
  }
  return path_separator;
}

/** 
 * Returns the path as a string with the appropriate
 * path separator.
 * 
 * @param model_path vector of strings representing path to the model
 * 
 * @return the string representation of the path with the appropriate
 *    path separator.
 */
std::string convert_model_path(const std::vector<std::string>& model_path) {
  std::string path;
  if (model_path.size() > 0) {
    path.append(model_path[0]);
    for (size_t i = 1; i < model_path.size(); i++) {
      path.append(1, get_path_separator());
      path.append(model_path[i]);
    }
  }
  return path;
}

struct run_command_output {
  std::string command;
  std::string output;
  long time;
  int err_code;
  bool hasError;

  run_command_output(const std::string command,
                     const std::string output,
                     const long time,
                     const int err_code)
    : command(command),
      output(output),
      time(time),
      err_code(err_code),
      hasError(err_code != 0)
  { }
  
  run_command_output() 
    : command(),
      output(),
      time(0),
      err_code(0),
      hasError(false)
      { }
};

std::ostream& operator<<(std::ostream& os, const run_command_output& out) {
  os << "run_command output:" << std::endl
     << "  command:   " << out.command << std::endl
     << "  output:    " << out.output << std::endl
     << "  time (ms): " << out.time << std::endl
     << "  err_code:  " << out.err_code << std::endl
     << "  hasError:  " << (out.hasError ? "true" : "false") << std::endl;
  return os;
}

/** 
 * Runs the command provided and returns the system output
 * as a string.
 * 
 * @param command A command that can be run from the shell
 * @return the system output of the command
 */  
run_command_output run_command(std::string command) {
  using boost::posix_time::ptime;
  using boost::posix_time::microsec_clock;
  
  FILE *in;
  std::string new_command = command + " 2>&1"; 
  // captures both std::cout amd std::err
  
  in = popen(command.c_str(), "r");
  
  if(!in) {
    std::string err_msg;
    err_msg = "Fatal error with popen; could not execute: \"";
    err_msg+= command;
    err_msg+= "\"";
    throw std::runtime_error(err_msg.c_str());
  }
  
  std::string output;
  char buf[1024];
  size_t count;
  ptime time_start(microsec_clock::universal_time()); // start timer
  while ((count = fread(&buf, 1, 1024, in)) > 0)
    output += std::string(&buf[0], &buf[count]);
  ptime time_end(microsec_clock::universal_time());   // end timer

  // bits 15-8 is err code, bit 7 if core dump, bits 6-0 is signal number
  int err_code = pclose(in);
  // on Windows, err code is the return code.
  if (err_code != 0 && (err_code >> 8) > 0)
    err_code >>= 8;

  return run_command_output(command, output,
                            (time_end - time_start).total_milliseconds(), 
                            err_code);
}

/** 
 * Returns the help options from the string provided.
 * Help options start with "--".
 * 
 * @param help_output output from "model/command --help"
 * @return a vector of strings of the help options
 */
std::vector<std::string> parse_help_options(const std::string& help_output) {
  std::vector<std::string> help_options;
  
  size_t option_start = help_output.find("--");
  while (option_start != std::string::npos) {
    // find the option name (skip two characters for "--")
    option_start += 2;
    size_t option_end = help_output.find_first_of("= ", option_start);
    help_options.push_back(help_output.substr(option_start, option_end-option_start));
    option_start = help_output.find("--", option_start+1);
  }
        
  return help_options;
}

/** 
 * Parses output from a Stan model run from the command line.
 * Returns option, value pairs.
 * 
 * @param command_output The output from a Stan model run from the command line.
 * 
 * @return Option, value pairs as indicated by the Stan model.
 */
std::vector<std::pair<std::string, std::string> > 
parse_command_output(const std::string& command_output) {
  using std::vector;
  using std::pair;
  using std::string;
  vector<pair<string, string> > output;
  
  string option, value;
  size_t start = 0, end = command_output.find("\n", start);
  
  start = end+1;
  end = command_output.find("\n", start);
  size_t equal_pos = command_output.find("=", start);
  
  while (equal_pos != string::npos) {
    using boost::trim;
    option = command_output.substr(start, equal_pos-start);
    value = command_output.substr(equal_pos+1, end - equal_pos - 1);
    trim(option);
    trim(value);
    output.push_back(pair<string, string>(option, value));
    start = end+1;
    end = command_output.find("\n", start);
    equal_pos = command_output.find("=", start);
  }
  return output;
}

#endif
