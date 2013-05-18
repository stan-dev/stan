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


/** 
 * Runs the command provided and returns the system output
 * as a string.
 * 
 * @param command A command that can be run from the shell
 * @return the system output of the command
 */  
std::string run_command(std::string command) {
  FILE *in;
  command += " 2>&1"; // capture stderr
  if(!(in = popen(command.c_str(), "r"))) {
    std::string err_msg;
    err_msg = "Could not run: \"";
    err_msg+= command;
    err_msg+= "\"";
    throw std::runtime_error(err_msg.c_str());
  }
  
  std::string output;
  char buf[1024];
  size_t count;
  while ((count = fread(&buf, 1, 1024, in)) > 0)
    output += std::string(&buf[0], &buf[count]);
  
  int err;
  if ((err=pclose(in)) != 0) {
    std::stringstream err_msg;
    err_msg << "Run of command: \"" << command << std::endl;
    err_msg << "err code: " << err << std::endl;
    err_msg << "Output message: \n";
    err_msg << output;
    std::string msg(err_msg.str());
    throw std::runtime_error(msg.c_str());
  }

  return output;
}

/** 
 * Runs the command provided and returns the system output
 * as a string.
 * 
 * @param[in] command A command that can be run from the shell
 * @param[out] elapsed_milliseconds Adds number of milliseconds run to current value.
 * @return the system output of the command
 */  
std::string run_command(const std::string& command, long& elapsed_milliseconds) {
  using boost::posix_time::ptime;
  using boost::posix_time::microsec_clock;
  
  ptime time_start(microsec_clock::universal_time()); // start timer
  std::string output = run_command(command);
  ptime time_end(microsec_clock::universal_time());   // end timer

  elapsed_milliseconds += (time_end - time_start).total_milliseconds();
  
  return output;
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
  
  EXPECT_EQ("STAN SAMPLING COMMAND", 
            command_output.substr(start, end))
    << "command could not be run. output is: \n" 
    << command_output;
  if ("STAN SAMPLING COMMAND" != command_output.substr(start, end)) {
    return output;
  }
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
