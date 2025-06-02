#ifndef STAN_RUN_READ_JSON_DATA_HPP
#define STAN_RUN_READ_JSON_DATA_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/empty_var_context.hpp>

#include <memory>
#include <string>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace stan {
namespace run {

/* Read JSON data from a file and return it as a var_context.
 *
 * @param filename Path to the JSON file
 * @return std::shared_ptr to a var_context containing the parsed JSON data
 * @throws std::runtime_error if the file cannot be opened
 */
inline std::shared_ptr<stan::io::var_context> read_json_data(const std::string& filename) {
  if (filename.empty()) {
    return std::make_shared<stan::io::empty_var_context>();
  }
    
  std::ifstream in(filename);
  if (!in) {
    if (!std::filesystem::exists(filename)) {
      throw std::runtime_error("Data file does not exist: " + filename);
    } else {
      throw std::runtime_error("Could not open data file (permission denied?): " + filename);
    }
  }
  return std::make_shared<stan::json::json_data>(in);
}

}  // namespace run
}  // namespace stan

#endif
