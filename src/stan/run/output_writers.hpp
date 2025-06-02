#ifndef STAN_RUN_OUTPUT_WRITERS_HPP
#define STAN_RUN_OUTPUT_WRITERS_HPP

#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>

namespace stan {
namespace run {

// Convenience aliases for commonly used writer types
using csv_writer = stan::callbacks::unique_stream_writer<std::ofstream>;
using json_writer = stan::callbacks::json_writer<std::ofstream>;

/**
 * Generate a timestamp string in format YYYYMMDD_HHMMSS
 */
inline std::string generate_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&time_t);
  
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

/**
 * Generate filename for output files
 * 
 * @param model_name Name of the Stan model
 * @param timestamp Timestamp string
 * @param chain_id Chain number (1-indexed)
 * @param data_type Type of data ("sample", "start_params", "param_grads", "metric")
 * @param extension File extension (".csv" or ".json")
 * @return Complete filename
 */
inline std::string generate_filename(const std::string& model_name,
                             const std::string& timestamp, 
                             unsigned int chain_id,
                             const std::string& data_type,
                             const std::string& extension) {
  return model_name + "_" + timestamp + "_chain" + std::to_string(chain_id) 
         + "_" + data_type + extension;
}

/**
 * Create output directory if it doesn't exist
 * 
 * @param output_dir Path to output directory
 * @throws std::runtime_error if directory cannot be created
 */
inline void ensure_output_directory(const std::string& output_dir) {
  if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
    if (!std::filesystem::create_directories(output_dir)) {
      throw std::runtime_error("Failed to create output directory: " + output_dir);
    }
  }
}

/**
 * Create complete file path
 * 
 * @param output_dir Output directory path
 * @param filename Filename
 * @return Complete file path
 */
inline std::string create_file_path(const std::string& output_dir, 
                            const std::string& filename) {
  if (output_dir.empty()) {
    return filename;
  }
  return std::filesystem::path(output_dir) / filename;
}

// Type traits to detect writer types
namespace traits {
  template <typename T>
  struct is_stream_writer : std::false_type {};
  
  template <typename Stream, typename Deleter>
  struct is_stream_writer<stan::callbacks::unique_stream_writer<Stream, Deleter>> : std::true_type {};
  
  template <typename T>
  struct is_json_writer : std::false_type {};
  
  template <typename Stream, typename Deleter>
  struct is_json_writer<stan::callbacks::json_writer<Stream, Deleter>> : std::true_type {};
}

/**
 * Create writer helper function for stream writers
 */
template <typename WriterType>
typename std::enable_if<traits::is_stream_writer<WriterType>::value, 
                       std::unique_ptr<WriterType>>::type
inline create_writer_impl(const std::string& filepath, const std::string& comment_prefix) {
  auto stream = std::make_unique<std::ofstream>(filepath);
  if (!stream->is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }
  return std::make_unique<WriterType>(std::move(stream), comment_prefix);
}

/**
 * Create writer helper function for JSON writers
 */
template <typename WriterType>
typename std::enable_if<traits::is_json_writer<WriterType>::value, 
                       std::unique_ptr<WriterType>>::type
inline create_writer_impl(const std::string& filepath, const std::string&) {
  auto stream = std::make_unique<std::ofstream>(filepath);
  if (!stream->is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }
  return std::make_unique<WriterType>(std::move(stream));
}

/**
 * Generic function to create any type of writer
 * 
 * @param output_dir Output directory path
 * @param model_name Name of the Stan model
 * @param timestamp Timestamp string
 * @param chain_id Chain number (1-indexed)
 * @param data_type Type of data (e.g., "sample", "metric")
 * @param extension File extension (e.g., ".csv", ".json")
 * @param comment_prefix Optional comment prefix (only used for stream writers)
 * @return Unique pointer to the requested writer type
 */
template <typename WriterType>
std::unique_ptr<WriterType>
inline create_writer(const std::string& output_dir,
             const std::string& model_name,
             const std::string& timestamp,
             unsigned int chain_id,
             const std::string& data_type,
             const std::string& extension,
             const std::string& comment_prefix = "") {
  std::string filename = generate_filename(model_name, timestamp, chain_id, 
                                          data_type, extension);
  std::string filepath = create_file_path(output_dir, filename);
  return create_writer_impl<WriterType>(filepath, comment_prefix);
}

}  // namespace run
}  // namespace stan
#endif
