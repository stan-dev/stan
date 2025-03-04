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

  // Get structured writer for backward compatibility with existing code
  template<typename Stream = std::ofstream, typename Deleter = std::default_delete<Stream>>
  callbacks::structured_writer* get_structured_writer(const std::string& info_type) {
    auto writer = get_writer(info_type);
    auto* structured_writer = dynamic_cast<callbacks::structured_writer*>(writer.get());
    if (!structured_writer) {
      throw std::runtime_error("Writer for " + info_type + " is not a structured writer");
    }
    return structured_writer;
  }

  // Get JSON writer for backward compatibility with existing code
  template<typename Stream = std::ofstream, typename Deleter = std::default_delete<Stream>>
  callbacks::json_writer<Stream, Deleter>* get_json_writer(const std::string& info_type) {
    auto writer = get_writer(info_type);
    auto* json_writer = dynamic_cast<callbacks::json_writer<Stream, Deleter>*>(writer.get());
    if (!json_writer) {
      throw std::runtime_error("Writer for " + info_type + " is not a JSON writer");
    }
    return json_writer;
  }

  // Get Arrow writer for backward compatibility with existing code
  template<typename Stream = std::ofstream, typename Deleter = std::default_delete<Stream>>
  callbacks::arrow_writer<Stream, Deleter>* get_arrow_writer(const std::string& info_type) {
    auto writer = get_writer(info_type);
    auto* arrow_writer = dynamic_cast<callbacks::arrow_writer<Stream, Deleter>*>(writer.get());
    if (!arrow_writer) {
      throw std::runtime_error("Writer for " + info_type + " is not an Arrow writer");
    }
    return arrow_writer;
  }

 private:
  std::map<std::string, OutputConfig> output_configs_;
  
  std::unique_ptr<callbacks::writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV:
        return std::make_unique<callbacks::stream_writer>(config.path);
      case OutputFormat::MATRIX:
        if (config.rows == 0 || config.cols == 0) {
          throw std::runtime_error("Matrix dimensions must be specified");
        }
        return std::make_unique<callbacks::matrix_writer>(config.rows, config.cols);
      case OutputFormat::ARROW: {
        auto file = std::make_unique<std::ofstream>(config.path);
        return std::make_unique<callbacks::structured_writer<std::ofstream>>(std::move(file));
      }
      case OutputFormat::JSON: {
        auto file = std::make_unique<std::ofstream>(config.path);
        return std::make_unique<callbacks::json_writer<std::ofstream>>(std::move(file));
      }
      default:
        throw std::runtime_error("Invalid output format");
    }
  }
};

} // namespace io
} // namespace stan

#endif 