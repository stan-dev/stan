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

#endif
