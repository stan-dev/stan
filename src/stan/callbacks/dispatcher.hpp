#ifndef STAN_CALLBACKS_DISPATCHER_HPP
#define STAN_CALLBACKS_DISPATCHER_HPP


#include <iostream>
#include <memory> // for std::shared_ptr
#include <map>
#include <string>
#include <utility>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/structured_writer.hpp>

namespace stan {
namespace callbacks {

/**
 * <code>dispatcher</code> is a base class defining the interface
 * for Stan dispatcher callbacks. The base class can be used as a
 * no-op implementation.
 */
class dispatcher {
 private:
  std::map<info_type, std::shared_ptr<structured_writer>> writers_;

 public:
  /** default constructor */
  dispatcher() {}

  /** copy constructor */
  dispatcher(dispatcher& other) = delete;

  /** move constructor */
  dispatcher(dispatcher&& other) noexcept
      : writers_(std::move(other.writers_)) {}

  /** virtual destructor */
  virtual ~dispatcher() {}
  
  /**
   * Add mapping from info_type to writer
   */
  void add_writer(const info_type& info, std::shared_ptr<structured_writer>&& writer) {
    writers_[info] = writer;
  }

  template <typename First, typename... Rest>
  void write(First&& first, Rest&&... rest) {
    auto info_type = std::forward<First>(first);
    if (writers_.find(info_type) != writers_.end())
      writers_[info_type]->write(std::forward<Rest>(rest)...);
  }

  template <typename First, typename... Rest>
  void begin_record(First&& first, Rest&&... rest) {
    auto info_type = std::forward<First>(first);
    if (writers_.find(info_type) != writers_.end())
      writers_[info_type]->begin_record(std::forward<Rest>(rest)...);
  }

  template <typename First, typename... Rest>
  void end_record(First&& first, Rest&&... rest) {
    auto info_type = std::forward<First>(first);
    if (writers_.find(info_type) != writers_.end())
      writers_[info_type]->end_record(std::forward<Rest>(rest)...);
  }

  template <typename First, typename... Rest>
  void table_header(First&& first, Rest&&... rest) {
    auto info_type = std::forward<First>(first);
    if (writers_.find(info_type) != writers_.end())
      writers_[info_type]->table_header(std::forward<Rest>(rest)...);
  }

  template <typename First, typename... Rest>
  void table_row(First&& first, Rest&&... rest) {
    auto info_type = std::forward<First>(first);
    if (writers_.find(info_type) != writers_.end())
      writers_[info_type]->table_row(std::forward<Rest>(rest)...);
  }

};


}  // namespace callbacks
}  // namespace stan
#endif

