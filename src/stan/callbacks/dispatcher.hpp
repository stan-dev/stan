#ifndef STAN_CALLBACKS_DISPATCHER_HPP
#define STAN_CALLBACKS_DISPATCHER_HPP


#include <iostream>
#include <memory> // for std::shared_ptr
#include <map>
#include <string>
#include <utility>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/writer.hpp>

namespace stan {
namespace callbacks {

class dispatcher;  // Forward declaration

template <typename EnumType>
struct WriterMapAccessor;

/**
 * <code>dispatcher</code> is a base class defining the interface
 * for Stan dispatcher callbacks. The base class can be used as a
 * no-op implementation.
 */
class dispatcher {

 private:
  std::map<table_info_type, std::shared_ptr<structured_writer>> table_writers_;
  std::map<struct_info_type, std::shared_ptr<structured_writer>> struct_writers_;
  std::map<stream_info_type, std::shared_ptr<writer>> stream_writers_;
  
  auto find_writer_impl(const table_info_type& info) {
    auto it = table_writers_.find(info);
    if (it == table_writers_.end()) {
      return std::shared_ptr<structured_writer>(nullptr);
    }
    return it->second;
  }

  auto find_writer_impl(const struct_info_type& info) {
    auto it = struct_writers_.find(info);
    if (it == struct_writers_.end()) {
      return std::shared_ptr<structured_writer>(nullptr);
    }
    return it->second;
  }

  auto find_writer_impl(const stream_info_type& info) {
    auto it = stream_writers_.find(info);
    if (it == stream_writers_.end()) {
      return std::shared_ptr<writer>(nullptr);
    }
    return it->second;
  }

 public:
  template <typename EnumType>
  friend struct WriterMapAccessor;

  /** default constructor */
  dispatcher() {}

  /** copy constructor */
  dispatcher(dispatcher& other) = delete;

  /** move constructor */
  dispatcher(dispatcher&& other) noexcept
      : table_writers_(std::move(other.table_writers_)),
        struct_writers_(std::move(other.struct_writers_)),
        stream_writers_(std::move(other.stream_writers_)) {
  }

  /** virtual destructor */
  virtual ~dispatcher() {}
  
  /**
   * Add mapping from info_type to writer
   */
  void add_writer(const table_info_type& info, std::shared_ptr<structured_writer>&& writer) {
    table_writers_[info] = writer;
  }

  void add_writer(const struct_info_type& info, std::shared_ptr<structured_writer>&& writer) {
    struct_writers_[info] = writer;
  }

  void add_writer(const stream_info_type& info, std::shared_ptr<writer>&& writer) {
    stream_writers_[info] = writer;
  }

  
  template <typename First, typename... Rest>
  void write(First&& first, Rest&&... rest) {
    auto& map = WriterMapAccessor<std::decay_t<First>>::get_map(*this);
    auto it = map.find(std::forward<First>(first));
    if (it != map.end()) {
      it->second->write(std::forward<Rest>(rest)...);
    }
  }

  template <typename First, typename... Rest>
  void begin_record(First&& first, Rest&&... rest) {
    auto& map = WriterMapAccessor<std::decay_t<First>>::get_map(*this);
    auto it = map.find(std::forward<First>(first));
    if (it != map.end()) {
      it->second->begin_record(std::forward<Rest>(rest)...);
    }
  }
  
  template <typename First, typename... Rest>
  void end_record(First&& first, Rest&&... rest) {
    auto& map = WriterMapAccessor<std::decay_t<First>>::get_map(*this);
    auto it = map.find(std::forward<First>(first));
    if (it != map.end()) {
      it->second->end_record(std::forward<Rest>(rest)...);
    }
  }
    
  template <typename First, typename... Rest>
  void table_header(First&& first, Rest&&... rest) {
    auto& map = WriterMapAccessor<std::decay_t<First>>::get_map(*this);
    auto it = map.find(std::forward<First>(first));
    if (it != map.end()) {
      it->second->table_header(std::forward<Rest>(rest)...);
    }
  }

  template <typename First, typename... Rest>
  void table_row(First&& first, Rest&&... rest) {
    auto& map = WriterMapAccessor<std::decay_t<First>>::get_map(*this);
    auto it = map.find(std::forward<First>(first));
    if (it != map.end()) {
      it->second->table_row(std::forward<Rest>(rest)...);
    }
  }

};

// specializations of WriterMapAccessor for info type maps

template <>
struct WriterMapAccessor<table_info_type> {
    static auto get_map(dispatcher& d) -> decltype(d.table_writers_)& {
        return d.table_writers_;
    }
};

template <>
struct WriterMapAccessor<struct_info_type> {
    static auto get_map(dispatcher& d) -> decltype(d.struct_writers_)& {
        return d.struct_writers_;
    }
};

template <>
struct WriterMapAccessor<stream_info_type> {
    static auto get_map(dispatcher& d) -> decltype(d.stream_writers_)& {
        return d.stream_writers_;
    }
};

}  // namespace callbacks
}  // namespace stan
#endif

