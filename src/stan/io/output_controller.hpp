#ifndef STAN_IO_OUTPUT_CONTROLLER_HPP
#define STAN_IO_OUTPUT_CONTROLLER_HPP

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
namespace io {

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
  std::unordered_map<std::string, std::shared_ptr<callbacks::writer>> writers_;

  std::shared_ptr<callbacks::writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV: {
        auto file = std::make_unique<std::ofstream>(config.file_path);
        return std::make_shared<callbacks::stream_writer>(*file);
      }
      case OutputFormat::MATRIX:
        return std::make_shared<callbacks::matrix_writer>(
            config.dims.rows, config.dims.cols);
      case OutputFormat::ARROW:
        // TODO: Implement Arrow writer
        throw std::runtime_error("Arrow writer not yet implemented");
      case OutputFormat::JSON: {
        auto file = std::make_unique<std::ofstream>(config.file_path);
        auto json_writer = std::make_shared<callbacks::json_writer<std::ofstream, std::default_delete<std::ofstream>>>(
            std::move(file));
        return std::dynamic_pointer_cast<callbacks::writer>(json_writer);
      }
      default:
        throw std::runtime_error("Invalid output format");
    }
  }

 public:
  void configure_output(const std::string& info_type, const OutputConfig& config) {
    writers_[info_type] = create_writer(config);
  }

  std::shared_ptr<callbacks::writer> get_writer(const std::string& info_type) {
    auto it = writers_.find(info_type);
    if (it == writers_.end()) {
      throw std::runtime_error("No writer configured for " + info_type);
    }
    return it->second;
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

  void write(const std::string& info_type, const std::vector<double>& data) {
    auto writer = get_writer(info_type);
    writer->operator()(data);
  }

  void write_metadata(const std::string& info_type, const std::vector<std::string>& metadata) {
    auto writer = get_writer(info_type);
    writer->operator()(metadata);
  }
};

} // namespace io
} // namespace stan

#endif 