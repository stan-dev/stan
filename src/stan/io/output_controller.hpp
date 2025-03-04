#ifndef STAN_IO_OUTPUT_CONTROLLER_HPP
#define STAN_IO_OUTPUT_CONTROLLER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/matrix_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <fstream>

namespace stan {
namespace io {

enum class OutputFormat {
  CSV,      // Plain text CSV files for streaming data
  MATRIX,   // In-memory column-major matrix
  ARROW,    // Apache Arrow format for streaming data
  JSON      // JSON format for flexible metadata
};

struct OutputConfig {
  OutputFormat format;
  std::string path;  // File path or identifier for the output
  size_t rows{0};    // For matrix format
  size_t cols{0};    // For matrix format
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
  
  // Write streaming data (samples, diagnostics) - works for CSV, MATRIX, and ARROW
  void write(const std::string& info_type, const std::vector<double>& data) {
    auto writer = get_writer(info_type);
    writer->operator()(data);
  }

  // Write metadata (flexible JSON format)
  void write_metadata(const std::string& info_type, 
                     const std::vector<std::string>& metadata) {
    auto writer = get_writer(info_type);
    writer->operator()(metadata);
  }

  // For backward compatibility with existing JSON writer usage
  callbacks::json_writer* get_json_writer(const std::string& info_type) {
    auto writer = get_writer(info_type);
    auto* json_writer = dynamic_cast<callbacks::json_writer*>(writer.get());
    if (!json_writer) {
      throw std::runtime_error("Writer is not a JSON writer");
    }
    return json_writer;
  }

 private:
  std::map<std::string, OutputConfig> output_configs_;
  
  std::unique_ptr<callbacks::writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV: {
        auto* file = new std::ofstream(config.path);
        return std::unique_ptr<callbacks::writer>(
            new callbacks::stream_writer(*file));
      }
      case OutputFormat::MATRIX:
        if (config.rows == 0 || config.cols == 0) {
          throw std::runtime_error("Matrix dimensions must be specified");
        }
        return std::make_unique<callbacks::matrix_writer>(config.rows, config.cols);
      case OutputFormat::ARROW: {
        auto* file = new std::ofstream(config.path);
        return std::unique_ptr<callbacks::writer>(
            new callbacks::structured_writer(*file));
      }
      case OutputFormat::JSON: {
        auto* file = new std::ofstream(config.path);
        return std::unique_ptr<callbacks::writer>(
            new callbacks::json_writer(*file));
      }
      default:
        throw std::runtime_error("Unsupported output format");
    }
  }
};

} // namespace io
} // namespace stan

#endif 