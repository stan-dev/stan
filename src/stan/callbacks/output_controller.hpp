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
  std::unordered_map<std::string, std::shared_ptr<structured_writer>> structured_writers_;
  std::unordered_map<std::string, std::shared_ptr<std::ofstream>> files_;

  std::shared_ptr<writer> create_writer(const OutputConfig& config) {
    switch (config.format) {
      case OutputFormat::CSV: {
        auto file = std::make_shared<std::ofstream>(config.file_path);
        files_[config.file_path] = file;
        return std::make_shared<stream_writer>(*file);
      }
      case OutputFormat::MATRIX:
        return std::make_shared<matrix_writer>(
            config.dims.rows, config.dims.cols);
      case OutputFormat::JSON: {
        auto file = std::make_shared<std::ofstream>(config.file_path);
        files_[config.file_path] = file;
        auto file_ptr = std::make_unique<std::ofstream>(config.file_path);
        auto jwriter = std::make_shared<json_writer<std::ofstream>>(
            std::move(file_ptr));
        structured_writers_[config.file_path] = jwriter;
        return nullptr;  // JSON writers are handled separately
      }
      default:
        throw std::runtime_error("Invalid output format");
    }
  }

 public:
  void configure_output(const std::string& info_type, const OutputConfig& config) {
    auto writer = create_writer(config);
    if (writer) {
      writers_[info_type] = writer;
    }
  }

  std::shared_ptr<writer> get_writer(const std::string& info_type) {
    auto it = writers_.find(info_type);
    if (it == writers_.end()) {
      throw std::runtime_error("No writer configured for " + info_type);
    }
    return it->second;
  }

  template<typename Stream = std::ofstream>
  json_writer<Stream>* get_json_writer(const std::string& info_type) {
    auto it = structured_writers_.find(info_type);
    if (it == structured_writers_.end()) {
      throw std::runtime_error("No JSON writer configured for " + info_type);
    }
    return dynamic_cast<json_writer<Stream>*>(it->second.get());
  }

  void write(const std::string& info_type, const std::vector<double>& data) {
    auto writer = get_writer(info_type);
    writer->operator()(data);
  }
};

} // namespace callbacks
} // namespace stan

#endif 
