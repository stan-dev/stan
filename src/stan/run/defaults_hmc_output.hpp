#ifndef STAN_RUN_OUTPUT_DEFAULTS_HPP
#define STAN_RUN_OUTPUT_DEFAULTS_HPP

#include <stan/run/config.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <string>
#include <memory>

namespace stan {
namespace run {

/**
 * Default initialization writer configuration.
 */
template <typename Stream = std::ostream, typename Deleter = std::default_delete<Stream>>
class init_writer_config : public config<callbacks::unique_stream_writer<Stream, Deleter>*> {
private:
  static const std::string description_;
  static void validator(const callbacks::unique_stream_writer<Stream, Deleter>* const& value) {
    // No validation needed - nullptr is a valid value
  }

public:
  init_writer_config() : config<callbacks::unique_stream_writer<Stream, Deleter>*>(nullptr, description_, validator) {}
  
  explicit init_writer_config(callbacks::unique_stream_writer<Stream, Deleter>* value) 
    : config<callbacks::unique_stream_writer<Stream, Deleter>*>(value, description_, validator) {}

  static callbacks::unique_stream_writer<Stream, Deleter>* default_value() { return nullptr; }
};

template <typename Stream, typename Deleter>
const std::string init_writer_config<Stream, Deleter>::description_ = "Writer for parameter initialization values.";

/**
 * Default sample writer configuration.
 */
template <typename Stream = std::ostream, typename Deleter = std::default_delete<Stream>>
class sample_writer_config : public config<callbacks::unique_stream_writer<Stream, Deleter>*> {
private:
  static const std::string description_;
  static void validator(const callbacks::unique_stream_writer<Stream, Deleter>* const& value) {
    // No validation needed - nullptr is a valid value
  }

public:
  sample_writer_config() : config<callbacks::unique_stream_writer<Stream, Deleter>*>(nullptr, description_, validator) {}
  
  explicit sample_writer_config(callbacks::unique_stream_writer<Stream, Deleter>* value) 
    : config<callbacks::unique_stream_writer<Stream, Deleter>*>(value, description_, validator) {}

  static callbacks::unique_stream_writer<Stream, Deleter>* default_value() { return nullptr; }
};

template <typename Stream, typename Deleter>
const std::string sample_writer_config<Stream, Deleter>::description_ = "Writer for MCMC samples.";

/**
 * Default diagnostic writer configuration.
 */
template <typename Stream = std::ostream, typename Deleter = std::default_delete<Stream>>
class diagnostic_writer_config : public config<callbacks::unique_stream_writer<Stream, Deleter>*> {
private:
  static const std::string description_;
  static void validator(const callbacks::unique_stream_writer<Stream, Deleter>* const& value) {
    // No validation needed - nullptr is a valid value
  }

public:
  diagnostic_writer_config() : config<callbacks::unique_stream_writer<Stream, Deleter>*>(nullptr, description_, validator) {}
  
  explicit diagnostic_writer_config(callbacks::unique_stream_writer<Stream, Deleter>* value) 
    : config<callbacks::unique_stream_writer<Stream, Deleter>*>(value, description_, validator) {}

  static callbacks::unique_stream_writer<Stream, Deleter>* default_value() { return nullptr; }
};

template <typename Stream, typename Deleter>
const std::string diagnostic_writer_config<Stream, Deleter>::description_ = "Writer for diagnostic information.";

/**
 * Default metric writer configuration.
 */
template <typename Stream = std::ostream, typename Deleter = std::default_delete<Stream>>
class metric_writer_config : public config<callbacks::json_writer<Stream, Deleter>*> {
private:
  static const std::string description_;
  static void validator(const callbacks::json_writer<Stream, Deleter>* const& value) {
    // No validation needed - nullptr is a valid value
  }

public:
  metric_writer_config() : config<callbacks::json_writer<Stream, Deleter>*>(nullptr, description_, validator) {}
  
  explicit metric_writer_config(callbacks::json_writer<Stream, Deleter>* value) 
    : config<callbacks::json_writer<Stream, Deleter>*>(value, description_, validator) {}

  static callbacks::json_writer<Stream, Deleter>* default_value() { return nullptr; }
};

template <typename Stream, typename Deleter>
const std::string metric_writer_config<Stream, Deleter>::description_ = "Structured writer for sampler metrics.";

}  // namespace run
}  // namespace stan

#endif
