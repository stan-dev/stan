#ifndef STAN_CALLBACKS_DISPATCHER_HPP
#define STAN_CALLBACKS_DISPATCHER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <variant>

namespace stan {
namespace callbacks {

/**
 * The <code>dispatcher</code> class manages a set of callbacks
 * for all outputs of one run of a Stan service.
 * Calls to the dispatcher's `dispatch` method are forwarded to
 * the callback registered on the channel.
 */

/**
 * Enum `info_type` holds output type labels which are used by
 * the dispatcher class to map outputs to output channels.
 */
enum class info_type {
  CONFIG,      // series of string messages
  SAMPLE,      // draw from posterior
  SAMPLE_RAW,  // draw from posterior
  METRIC,      // struct with kv pairs 'metric_type', 'stepsize', 'inv_metric'
  ALGORITHM_STATE,     // sampler state for returned draw
  DIAGNOSTIC,          // parameter gradients
  UNCONSTRAINED_INITS  // unconstrained parameter values
};

/**
 * Efficient enum type lookups
 */
struct info_type_hash {
  std::size_t operator()(const info_type& type) const {
    return std::hash<int>()(static_cast<int>(type));
  }
};

/**
 * Base type for all callbacks, needed for type erasure.
 */
class channel {
 public:
  virtual ~channel() = default;
};

/**
 * A `writer_channel` holds a reference to a stan::callbacks::writer object
 * and forwards information to the writer's operator ().
 */
class writer_channel : public channel {
 private:
  stan::callbacks::writer* writer_;

 public:
  explicit writer_channel(stan::callbacks::writer* w) : writer_(w) {
    if (!w) {
      throw std::runtime_error("config error, null writer");
    }
  }

  // Handle all types that writer supports via operator()
  void dispatch() { (*writer_)(); }
  void dispatch(const std::string& value) { (*writer_)(value); }
  void dispatch(const std::vector<double>& value) { (*writer_)(value); }
  void dispatch(const std::vector<std::string>& value) { (*writer_)(value); }

  // Handle any Eigen Matrix type
  template <int R, int C>
  void dispatch(const Eigen::Matrix<double, R, C>& value) {
    (*writer_)(value);
  }

  // No key-value support for plain writers
  template <typename T>
  void dispatch(const std::string&, const T&) {}
};

/**
 * A `structured writer_channel` holds a reference to a
 * stan::callbacks::structured_writer object and forwards
 * information to the appropriate method.
 */
class structured_writer_channel : public channel {
 private:
  stan::callbacks::structured_writer* writer_;

 public:
  explicit structured_writer_channel(stan::callbacks::structured_writer* sw)
      : writer_(sw) {
    if (!sw)
      throw std::runtime_error("config error, null writer");
  }
  // Forward all key-value calls directly to the writer
  void dispatch(const std::string& key) { writer_->write(key); }
  // Perfect forwarding for any key-value pair
  template <typename T>
  void dispatch(const std::string& key, T&& value) {
    writer_->write(key, std::forward<T>(value));
  }
  void begin_record() { writer_->begin_record(); }
  void begin_record(const std::string& key) { writer_->begin_record(key); }
  void end_record() { writer_->end_record(); }
};

/**
 * The `dispatcher` class provides methods to register and find output channels
 * and overloads method `dispatch` which forwards outputs to callbacks.
 */
class dispatcher {
 private:
  /* Lookup registered channels for info_type.
   * Returns nullptr if no channel found.
   */
  template <typename channel_type>
  channel_type* find_channel(info_type type) {
    auto it = channels_.find(type);
    if (it == channels_.end())
      return nullptr;
    return dynamic_cast<channel_type*>(it->second.get());
  }

  std::unordered_map<info_type, std::unique_ptr<channel>, info_type_hash>
      channels_;

  // neccesary for proper handling of shared ptrs
  std::vector<std::shared_ptr<void>> managed_resources_;

 public:
  dispatcher() = default;

  dispatcher(const dispatcher&) = delete;
  dispatcher& operator=(const dispatcher&) = delete;

  dispatcher(dispatcher&& other) noexcept
      : channels_(std::move(other.channels_)),
        managed_resources_(std::move(other.managed_resources_)) {}

  dispatcher& operator=(dispatcher&& other) noexcept {
    if (this != &other) {
      channels_ = std::move(other.channels_);
      managed_resources_ = std::move(other.managed_resources_);
    }
    return *this;
  }

  ~dispatcher() = default;

  /**
   * Managed resources are kept alive for the lifetime of the dispatcher.
   *
   * @param resource Shared pointer to the resource to manage
   */
  void add_managed_resource(std::shared_ptr<void> resource) {
    managed_resources_.push_back(std::move(resource));
  }


  /**
   * Add channel to map.
   * Assumes a 1:1 mapping between info type and callback.
   *
   * @param[in] type The info_type to associate with the channel
   * @param[in] channel A unique_ptr to the channel to register
   */
  void register_channel(info_type type, std::unique_ptr<channel> channel) {
    channels_[type] = std::move(channel);
  }

  /**
   * Dispatches a no-argument call to the writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   */
  void dispatch(info_type type) {
    if (auto* wc = find_channel<writer_channel>(type))
      wc->dispatch();
  }

  /**
   * Dispatches a vector of doubles to the writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] value Vector of doubles to be written
   */
  void dispatch(info_type type, const std::vector<double>& value) {
    if (auto* wc = find_channel<writer_channel>(type))
      wc->dispatch(value);
  }

  /**
   * Dispatches a vector of strings to the writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] value Vector of strings to be written
   */
  void dispatch(info_type type, const std::vector<std::string>& value) {
    if (auto* wc = find_channel<writer_channel>(type))
      wc->dispatch(value);
  }

  /**
   * Dispatches an Eigen matrix to the writer associated with the given type.
   *
   * @tparam R Number of rows in the matrix (-1 for dynamic)
   * @tparam C Number of columns in the matrix (-1 for dynamic)
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] value Eigen matrix to be written
   */
  template <int R, int C>
  void dispatch(info_type type, const Eigen::Matrix<double, R, C>& value) {
    if (auto* wc = find_channel<writer_channel>(type))
      wc->dispatch(value);
  }

  /**
   * Dispatches a string to the writer associated with the given type.
   * For structured writers, the string is treated as a key.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] value String to be written
   */
  void dispatch(info_type type, const std::string& value) {
    if (auto* wc = find_channel<writer_channel>(type))
      wc->dispatch(value);
    else if (auto* sw = find_channel<structured_writer_channel>(type))
      sw->dispatch(value);  // (sic: actually the key part of k-v pair)
  }

  /**
   * Dispatches a key-value pair to the structured writer associated with the given type.
   *
   * @tparam T Type of the value to be written
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] key Key string for the key-value pair
   * @param[in] value Value to be written
   */
  template <typename T>
  void dispatch(info_type type, const std::string& key, T&& value) {
    if (auto* sw = find_channel<structured_writer_channel>(type))
      sw->dispatch(key, std::forward<T>(value));
  }

  /**
   * Begins a record in the structured writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   */
  void begin_record(info_type type) {
    if (auto* sw = find_channel<structured_writer_channel>(type))
      sw->begin_record();
  }

  /**
   * Begins a named record in the structured writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   * @param[in] key Name of the record to begin
   */
  void begin_record(info_type type, const std::string& key) {
    if (auto* sw = find_channel<structured_writer_channel>(type))
      sw->begin_record(key);
  }

  /**
   * Ends a record in the structured writer associated with the given type.
   *
   * @param[in] type The info_type identifying the channel to dispatch to
   */
  void end_record(info_type type) {
    if (auto* sw = find_channel<structured_writer_channel>(type))
      sw->end_record();
  }
};

}  // namespace callbacks
}  // namespace stan

#endif  // STAN_CALLBACKS_DISPATCHER_HPP
