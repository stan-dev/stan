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

#endif
