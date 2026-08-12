#ifndef STAN_UTIL_RING_BUFFER_HPP
#define STAN_UTIL_RING_BUFFER_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace stan {
namespace util {

/**
 * A fixed-capacity ring buffer. Once full, each push evicts the oldest
 * element. The backing storage is allocated once, up front, and never
 * grows or shrinks on push/evict: `push_back()` hands back a reference to
 * an already-constructed (and, once the buffer has wrapped around at least
 * once, already appropriately-sized) element for the caller to assign into,
 * so pushing an `Eigen`-typed element reuses its existing heap buffer
 * instead of allocating a new one.
 *
 * @tparam T element type
 */
template <typename T>
class ring_buffer {
 public:
  /**
   * Construct a ring buffer with a fixed capacity.
   *
   * @param capacity maximum number of elements retained at once
   */
  explicit ring_buffer(size_t capacity)
      : buf_(capacity),
        capacity_(capacity),
        start_(0),
        size_(0),
        last_idx_(0) {}

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  void clear() {
    start_ = 0;
    size_ = 0;
  }

  /**
   * Make the next slot available for writing, evicting the oldest element
   * if the buffer is already full, and return a reference to it. The
   * returned slot holds whatever was last stored there (or a
   * default-constructed `T` if this capacity has never been filled), ready
   * to be overwritten by assignment.
   */
  T& push_back() {
    size_t idx;
    if (size_ < capacity_) {
      idx = (start_ + size_) % capacity_;
      ++size_;
    } else {
      idx = start_;
      start_ = (start_ + 1 == capacity_) ? 0 : start_ + 1;
    }
    last_idx_ = idx;
    return buf_[idx];
  }

  /**
   * Push a value, evicting the oldest element if the buffer is full.
   * Assigns directly into the reused slot rather than constructing a
   * temporary, so an rvalue expression (e.g. an Eigen expression template)
   * is evaluated straight into the slot's existing storage.
   */
  template <typename U>
  void push_back(U&& value) {
    push_back() = std::forward<U>(value);
  }

  T& back() { return buf_[last_idx_]; }
  const T& back() const { return buf_[last_idx_]; }

  T& operator[](size_t i) { return buf_[(start_ + i) % capacity_]; }
  const T& operator[](size_t i) const { return buf_[(start_ + i) % capacity_]; }

  /**
   * Change the capacity, keeping the most-recently-pushed
   * `min(size(), new_capacity)` elements.
   */
  void rset_capacity(size_t new_capacity) {
    std::vector<T> new_buf(new_capacity);
    size_t keep = std::min(size_, new_capacity);
    size_t old_first = (start_ + (size_ - keep)) % capacity_;
    for (size_t i = 0; i < keep; ++i)
      new_buf[i] = buf_[(old_first + i) % capacity_];
    buf_.swap(new_buf);
    capacity_ = new_capacity;
    start_ = 0;
    size_ = keep;
    last_idx_ = keep == 0 ? 0 : keep - 1;
  }

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() : buf_(nullptr), capacity_(0), idx_(0), pos_(0) {}
    const_iterator(const std::vector<T>* buf, size_t capacity, size_t idx,
                   size_t pos)
        : buf_(buf), capacity_(capacity), idx_(idx), pos_(pos) {}

    reference operator*() const { return (*buf_)[idx_]; }
    pointer operator->() const { return &(*buf_)[idx_]; }

    const_iterator& operator++() {
      ++pos_;
      idx_ = (idx_ + 1 == capacity_) ? 0 : idx_ + 1;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++*this;
      return tmp;
    }
    const_iterator& operator--() {
      --pos_;
      idx_ = (idx_ == 0) ? capacity_ - 1 : idx_ - 1;
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --*this;
      return tmp;
    }

    bool operator==(const const_iterator& o) const { return pos_ == o.pos_; }
    bool operator!=(const const_iterator& o) const { return pos_ != o.pos_; }

   private:
    const std::vector<T>* buf_;
    size_t capacity_;
    size_t idx_;
    size_t pos_;
  };

  const_iterator begin() const {
    return const_iterator(&buf_, capacity_, start_, 0);
  }
  const_iterator end() const {
    return const_iterator(&buf_, capacity_, (start_ + size_) % capacity_,
                          size_);
  }

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

 private:
  std::vector<T> buf_;
  size_t capacity_;
  size_t start_;
  size_t size_;
  size_t last_idx_;
};

}  // namespace util
}  // namespace stan
#endif
