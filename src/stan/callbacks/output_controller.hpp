#ifndef STAN_CALLBACKS_OUTPUT_CONTROLLER_HPP
#define STAN_CALLBACKS_OUTPUT_CONTROLLER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/matrix_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <fstream>

namespace stan {
namespace callbacks {

enum class OutputFormat {
  CSV,      // Plain text CSV files for streaming data
  MATRIX,   // In-memory column-major matrix
  ARROW,    // Apache Arrow format for streaming data
  JSON      // JSON format for flexible metadata
};

struct OutputDims {
  size_t rows;
  size_t cols;
};

struct OutputConfig {
  OutputFormat format;
  std::string file_path;
  OutputDims dims;
};

class output_controller {
 private:
  std::unordered_map<std::string, std::shared_ptr<writer>> writers_;
  std::unordered_map<std::string, std::shared_ptr<std::ofstream>> files_;  // Keep files alive

  std::shared_ptr<writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV: {
        auto file = std::make_shared<std::ofstream>(config.file_path);
        files_[config.file_path] = file;  // Store file
        return std::make_shared<stream_writer>(*file);
      }
      case OutputFormat::MATRIX:
        return std::make_shared<matrix_writer>(
            config.dims.rows, config.dims.cols);
      case OutputFormat::ARROW:
        // TODO: Implement Arrow writer
        throw std::runtime_error("Arrow writer not yet implemented");
      case OutputFormat::JSON: {
        auto file = std::make_shared<std::ofstream>(config.file_path);
        files_[config.file_path] = file;  // Store file
        auto file_ptr = std::make_unique<std::ofstream>(config.file_path);
        auto json_writer = std::make_shared<json_writer<std::ofstream, std::default_delete<std::ofstream>>>(
            std::move(file_ptr));
        return std::static_pointer_cast<writer>(json_writer);
      }
      default:
        throw std::runtime_error("Invalid output format");
    }
  }

 public:
  void configure_output(const std::string& info_type, const OutputConfig& config) {
    writers_[info_type] = create_writer(config);
  }

  std::shared_ptr<writer> get_writer(const std::string& info_type) {
    auto it = writers_.find(info_type);
    if (it == writers_.end()) {
      throw std::runtime_error("No writer configured for " + info_type);
    }
    return it->second;
  }

  // Forward all JSON writer methods to the underlying writer
  template<typename Stream = std::ofstream, typename Deleter = std::default_delete<Stream>>
  json_writer<Stream, Deleter>* get_json_writer(const std::string& info_type) {
    auto writer = get_writer(info_type);
    auto* json_writer = dynamic_cast<json_writer<Stream, Deleter>*>(writer.get());
    if (!json_writer) {
      throw std::runtime_error("Writer for " + info_type + " is not a JSON writer");
    }
    return json_writer;
  }

  void write(const std::string& info_type, const std::vector<double>& data) {
    auto writer = get_writer(info_type);
    writer->operator()(data);
  }

  void write_metadata(const std::string& info_type, const std::vector<std::string>& metadata) {
    auto writer = get_writer(info_type);
    writer->operator()(metadata);
  }
};

} // namespace callbacks
} // namespace stan

#endif 
