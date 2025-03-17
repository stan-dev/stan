#ifndef STAN_RUN_OUTPUT_CONFIG_HPP
#define STAN_RUN_OUTPUT_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace stan {
namespace run {

/**
 * Configuration class for output channels in Stan.
 * 
 * This class organizes output streams for both single-chain and multi-chain
 * sampling runs. It handles loggers and writers for samples, diagnostics, 
 * initialization values, and metric parameters.
 * 
 * The writers are expected to be unique_stream_writer instances, and
 * the metric writer is expected to be a json_writer.
 */
template <typename Stream = std::ostream, 
          typename Deleter = std::default_delete<Stream>>
class output_config {
 public:
  // Define type aliases for our writers
  using writer_type = callbacks::unique_stream_writer<Stream, Deleter>;
  using metric_writer_type = callbacks::json_writer<Stream, Deleter>;
  
  /**
   * Default constructor for a single-chain configuration.
   * Sets all writers to nullptr. Users must set required writers
   * before using the configuration.
   */
  output_config() 
    : num_chains_(1),
      logger_(nullptr) {
  }

  /**
   * Constructor for a single-chain configuration.
   * Any writer can be null if not needed.
   *
   * @param logger Logger for messages
   * @param init_writer Writer for parameter initialization (or nullptr)
   * @param sample_writer Writer for MCMC samples (or nullptr)
   * @param diagnostic_writer Writer for diagnostic information (or nullptr)
   * @param metric_writer Structured writer for sampler metrics (or nullptr)
   */
  output_config(
      callbacks::logger* logger,
      writer_type* init_writer = nullptr,
      writer_type* sample_writer = nullptr,
      writer_type* diagnostic_writer = nullptr,
      metric_writer_type* metric_writer = nullptr)
    : num_chains_(1),
      logger_(logger) {
    
    if (init_writer) init_writers_.push_back(init_writer);
    if (sample_writer) sample_writers_.push_back(sample_writer);
    if (diagnostic_writer) diagnostic_writers_.push_back(diagnostic_writer);
    if (metric_writer) metric_writers_.push_back(metric_writer);
  }

  /**
   * Constructor for multi-chain configuration.
   * Vectors can contain nullptr values for chains that don't need that writer.
   *
   * @param num_chains Number of chains
   * @param logger Shared logger for all chains
   * @param init_writers Vector of initialization writers, one per chain
   * @param sample_writers Vector of sample writers, one per chain
   * @param diagnostic_writers Vector of diagnostic writers, one per chain
   * @param metric_writers Vector of metric writers, one per chain (or empty)
   * @throws std::invalid_argument if vector sizes don't match num_chains
   */
  output_config(
      size_t num_chains,
      callbacks::logger* logger,
      const std::vector<writer_type*>& init_writers,
      const std::vector<writer_type*>& sample_writers,
      const std::vector<writer_type*>& diagnostic_writers,
      const std::vector<metric_writer_type*>& metric_writers = std::vector<metric_writer_type*>())
    : num_chains_(num_chains),
      logger_(logger) {
      
    validate_writer_sizes(num_chains, init_writers.size(), 
                          sample_writers.size(), diagnostic_writers.size());

    init_writers_.resize(num_chains);
    sample_writers_.resize(num_chains);
    diagnostic_writers_.resize(num_chains);

    for (size_t i = 0; i < num_chains; ++i) {
      init_writers_[i] = init_writers[i];
      sample_writers_[i] = sample_writers[i];
      diagnostic_writers_[i] = diagnostic_writers[i];
    }
    
    if (!metric_writers.empty()) {
      if (metric_writers.size() != num_chains) {
        throw std::invalid_argument("Metric writer vector size must match num_chains");
      }
      
      metric_writers_.resize(num_chains);
      for (size_t i = 0; i < num_chains; ++i) {
        metric_writers_[i] = metric_writers[i];
      }
    }
  }

  /**
   * Factory method for creating a single-chain configuration.
   *
   * @param logger Logger for messages
   * @param init_writer Writer for parameter initialization (or nullptr)
   * @param sample_writer Writer for MCMC samples (or nullptr)
   * @param diagnostic_writer Writer for diagnostic information (or nullptr)
   * @param metric_writer Structured writer for sampler metrics (or nullptr)
   * @return output_config Configuration with specified parameters
   */
  static output_config create(
      callbacks::logger* logger,
      writer_type* init_writer = nullptr,
      writer_type* sample_writer = nullptr,
      writer_type* diagnostic_writer = nullptr,
      metric_writer_type* metric_writer = nullptr) {
    
    return output_config(logger, init_writer, sample_writer, 
                        diagnostic_writer, metric_writer);
  }

  /**
   * Factory method for creating a multi-chain configuration.
   *
   * @param num_chains Number of chains
   * @param logger Shared logger for all chains
   * @param init_writers Vector of initialization writers, one per chain
   * @param sample_writers Vector of sample writers, one per chain
   * @param diagnostic_writers Vector of diagnostic writers, one per chain
   * @param metric_writers Vector of metric writers, one per chain (or empty)
   * @return output_config Configuration for specified chains
   */
  static output_config create_multi(
      size_t num_chains,
      callbacks::logger* logger,
      const std::vector<writer_type*>& init_writers,
      const std::vector<writer_type*>& sample_writers,
      const std::vector<writer_type*>& diagnostic_writers,
      const std::vector<metric_writer_type*>& metric_writers = std::vector<metric_writer_type*>()) {
    
    return output_config(num_chains, logger, init_writers, 
                        sample_writers, diagnostic_writers, metric_writers);
  }

  /**
   * Check if this configuration is for multiple chains.
   * 
   * @return true if configured for multiple chains (num_chains > 1)
   */
  bool is_multichain() const {
    return num_chains_ > 1;
  }

  /**
   * Check if all required writers are configured.
   * 
   * @return true if logger and all required writers are set
   */
  bool is_valid() const {
    if (!logger_ || init_writers_.empty() || 
        sample_writers_.empty() || diagnostic_writers_.empty()) {
      return false;
    }

    for (size_t i = 0; i < num_chains_; ++i) {
      if (!init_writers_[i] || !sample_writers_[i] || !diagnostic_writers_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Check if the configuration has metric writers.
   * 
   * @return true if metric writers are configured
   */
  bool has_metric_writer() const {
    return !metric_writers_.empty() && 
           std::any_of(metric_writers_.begin(), metric_writers_.end(), 
                       [](auto* ptr) { return ptr != nullptr; });
  }

  /**
   * Set a writer for a single chain configuration.
   * This method checks that the configuration is not multi-chain.
   *
   * @param writer The writer to set
   * @param writers The vector of writers to modify
   * @throws std::logic_error if configuration is multi-chain
   */
  template <typename Writer>
  void set_single_writer(Writer* writer, std::vector<Writer*>& writers) {
    if (is_multichain()) {
      throw std::logic_error("Cannot use single writer setters with multi-chain configuration");
    }
    
    if (writers.empty()) {
      writers.push_back(writer);
    } else {
      writers[0] = writer;
    }
  }

  // Getters and setters for number of chains
  size_t num_chains() const { return num_chains_; }

  // Logger getters and setters
  callbacks::logger* logger() const { return logger_; }
  void set_logger(callbacks::logger* logger) { logger_ = logger; }

  // Single chain getters (throw if multi-chain)
  writer_type* init_writer() const { 
    if (is_multichain()) {
      throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
    }
    return init_writers_.empty() ? nullptr : init_writers_[0]; 
  }
  
  writer_type* sample_writer() const { 
    if (is_multichain()) {
      throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
    }
    return sample_writers_.empty() ? nullptr : sample_writers_[0]; 
  }
  
  writer_type* diagnostic_writer() const { 
    if (is_multichain()) {
      throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
    }
    return diagnostic_writers_.empty() ? nullptr : diagnostic_writers_[0]; 
  }
  
  metric_writer_type* metric_writer() const { 
    if (is_multichain()) {
      throw std::logic_error("Cannot use single writer getters with multi-chain configuration");
    }
    return metric_writers_.empty() ? nullptr : metric_writers_[0]; 
  }

  // Single chain setters (throw if multi-chain)
  void set_init_writer(writer_type* init_writer) { 
    set_single_writer(init_writer, init_writers_);
  }
  
  void set_sample_writer(writer_type* sample_writer) { 
    set_single_writer(sample_writer, sample_writers_);
  }
  
  void set_diagnostic_writer(writer_type* diagnostic_writer) { 
    set_single_writer(diagnostic_writer, diagnostic_writers_);
  }
  
  void set_metric_writer(metric_writer_type* metric_writer) { 
    set_single_writer(metric_writer, metric_writers_);
  }

  // Multi-chain getters
  const std::vector<writer_type*>& init_writers() const { return init_writers_; }
  const std::vector<writer_type*>& sample_writers() const { return sample_writers_; }
  const std::vector<writer_type*>& diagnostic_writers() const { return diagnostic_writers_; }
  const std::vector<metric_writer_type*>& metric_writers() const { return metric_writers_; }

  // Individual chain getters
  writer_type* init_writer(size_t chain_idx) const { 
    validate_chain_idx(chain_idx);
    return init_writers_[chain_idx]; 
  }
  
  writer_type* sample_writer(size_t chain_idx) const { 
    validate_chain_idx(chain_idx);
    return sample_writers_[chain_idx]; 
  }
  
  writer_type* diagnostic_writer(size_t chain_idx) const { 
    validate_chain_idx(chain_idx);
    return diagnostic_writers_[chain_idx]; 
  }
  
  metric_writer_type* metric_writer(size_t chain_idx) const { 
    validate_chain_idx(chain_idx);
    return chain_idx < metric_writers_.size() ? metric_writers_[chain_idx] : nullptr; 
  }

  // Individual chain setters
  void set_init_writer(size_t chain_idx, writer_type* init_writer) {
    validate_chain_idx(chain_idx);
    init_writers_[chain_idx] = init_writer;
  }
  
  void set_sample_writer(size_t chain_idx, writer_type* sample_writer) {
    validate_chain_idx(chain_idx);
    sample_writers_[chain_idx] = sample_writer;
  }
  
  void set_diagnostic_writer(size_t chain_idx, writer_type* diagnostic_writer) {
    validate_chain_idx(chain_idx);
    diagnostic_writers_[chain_idx] = diagnostic_writer;
  }
  
  void set_metric_writer(size_t chain_idx, metric_writer_type* metric_writer) {
    validate_chain_idx(chain_idx);
    if (metric_writers_.size() <= chain_idx) {
      metric_writers_.resize(num_chains_, nullptr);
    }
    metric_writers_[chain_idx] = metric_writer;
  }

 private:
  /**
   * Validates that the vector sizes match the number of chains.
   */
  template <typename... Sizes>
  void validate_writer_sizes(size_t num_chains, Sizes... sizes) const {
    std::vector<size_t> size_vec = {static_cast<size_t>(sizes)...};
    for (size_t size : size_vec) {
      if (size != num_chains) {
        throw std::invalid_argument("Writer vector sizes must match num_chains");
      }
    }
  }

  /**
   * Validates that a chain index is within bounds.
   */
  void validate_chain_idx(size_t chain_idx) const {
    if (chain_idx >= num_chains_) {
      throw std::out_of_range("Chain index out of range");
    }
  }

  size_t num_chains_;
  callbacks::logger* logger_;
  std::vector<writer_type*> init_writers_;
  std::vector<writer_type*> sample_writers_;
  std::vector<writer_type*> diagnostic_writers_;
  std::vector<metric_writer_type*> metric_writers_;
};

}  // namespace run
}  // namespace stan
#endif
