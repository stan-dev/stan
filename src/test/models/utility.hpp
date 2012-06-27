#ifndef __TEST__MODELS__UTILITY_HPP__
#define __TEST__MODELS__UTILITY_HPP__

#include <stdexcept>

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
std::string run_command(const std::string& command) {
  FILE *in;
  if(!(in = popen(command.c_str(), "r"))) {
    std::string err_msg;
    err_msg = "Could not run: \"";
    err_msg+= command;
    err_msg+= "\"";
    throw std::runtime_error(err_msg.c_str());
  }
  
  std::string output;
  char buf[1024];
  size_t count = fread(&buf, 1, 1024, in);
  while (count > 0) {
    output += std::string(&buf[0], &buf[count]);
    count = fread(&buf, 1, 1024, in);
  }
  pclose(in);

  return output;
}



#endif
