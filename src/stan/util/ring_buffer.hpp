#ifndef STAN_UTIL_RING_BUFFER_HPP
#define STAN_UTIL_RING_BUFFER_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stan {
namespace util {

/** Fixed-capacity buffer that overwrites its oldest element when full. */
template <typename T>
class ring_buffer {
 public:
  explicit ring_buffer(size_t capacity) : buf_(capacity) {
    if (capacity == 0) {
      throw std::domain_error("ring_buffer capacity must be > 0");
    }
  }

  size_t size() const { return size_; }
  size_t capacity() const { return buf_.size(); }

  void clear() {
    start_ = 0;
    size_ = 0;
  }

  void push_back() {
    if (size_ < capacity()) {
      ++size_;
    } else {
      start_ = (start_ + 1) % capacity();
    }
  }

  template <typename U>
  void push_back(U&& value) {
    push_back();
    back() = std::forward<U>(value);
  }

  T& back() { return (*this)[size_ - 1]; }

  T& operator[](size_t i) { return buf_[(start_ + i) % capacity()]; }
  const T& operator[](size_t i) const {
    return buf_[(start_ + i) % capacity()];
  }

  void rset_capacity(size_t new_capacity) {
    if (new_capacity == 0) {
      throw std::domain_error("ring_buffer capacity must be > 0");
    }
    if (new_capacity == capacity()) {
      return;
    }

    std::vector<T> new_buf(new_capacity);
    size_t keep = std::min(size_, new_capacity);
    for (size_t i = 0; i < keep; ++i) {
      new_buf[i] = std::move((*this)[size_ - keep + i]);
    }
    buf_ = std::move(new_buf);
    start_ = 0;
    size_ = keep;
  }

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() = default;
    const_iterator(const ring_buffer* buffer, size_t pos)
        : buffer_(buffer), pos_(pos) {}

    reference operator*() const { return (*buffer_)[pos_]; }

    const_iterator& operator++() {
      ++pos_;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator result = *this;
      ++*this;
      return result;
    }
    const_iterator& operator--() {
      --pos_;
      return *this;
    }

    bool operator==(const const_iterator& other) const {
      return buffer_ == other.buffer_ && pos_ == other.pos_;
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    const ring_buffer* buffer_ = nullptr;
    size_t pos_ = 0;
  };

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, size_); }

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

 private:
  std::vector<T> buf_;
  size_t start_ = 0;
  size_t size_ = 0;
};

}  // namespace util
}  // namespace stan
#endif
