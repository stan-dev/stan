#ifndef STAN_IO_OUTPUT_CONTROLLER_HPP
#define STAN_IO_OUTPUT_CONTROLLER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace stan {
namespace io {

enum class OutputFormat {
  CSV,      // Plain text CSV files
  MATRIX,   // In-memory column-major matrix
  ARROW,    // Apache Arrow files
  JSON      // JSON files
};

struct OutputConfig {
  OutputFormat format;
  std::string path;  // File path or identifier for the output
};

class output_controller {
 public:
  output_controller() = default;
  
  // Configure output format for a specific information type
  void configure_output(const std::string& info_type, OutputConfig config) {
    output_configs_[info_type] = config;
  }
  
  // Get appropriate writer for information type
  std::unique_ptr<callbacks::writer> get_writer(const std::string& info_type) {
    auto it = output_configs_.find(info_type);
    if (it == output_configs_.end()) {
      throw std::runtime_error("No output configuration for " + info_type);
    }
    
    return create_writer(it->second);
  }
  
  // Write data using appropriate writer
  void write(const std::string& info_type, const std::vector<double>& data) {
    auto writer = get_writer(info_type);
    writer->operator()(data);
  }

 private:
  std::map<std::string, OutputConfig> output_configs_;
  
  std::unique_ptr<callbacks::writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV:
        return std::make_unique<stan::callbacks::stream_writer>(config.path);
      case OutputFormat::MATRIX:
        return std::make_unique<stan::callbacks::matrix_writer>(/* dimensions */);
      case OutputFormat::ARROW:
        return std::make_unique<stan::callbacks::arrow_writer>(config.path);
      case OutputFormat::JSON:
        return std::make_unique<stan::callbacks::json_writer>(config.path);
      default:
        throw std::runtime_error("Unsupported output format");
    }
  }
};

} // namespace io
} // namespace stan

#endif 