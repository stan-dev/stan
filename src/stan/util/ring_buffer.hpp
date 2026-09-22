#ifndef STAN_UTIL_RING_BUFFER_HPP
#define STAN_UTIL_RING_BUFFER_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {
namespace util {

/**
 * Fixed-capacity buffer that overwrites its oldest element when full.
 *
 * The interface follows the standard sequence container conventions where
 * they make sense for a ring: member typedefs, mutable and constant
 * random access iterators, reference qualified element access, and
 * conditionally `noexcept` operations that track the backing container.
 *
 * `data()` is deliberately absent because the elements are not contiguous
 * in logical order, and `allocator_type` is absent so that containers
 * without allocators may be used as the backing store.
 *
 * @tparam T Type of elements stored in the buffer
 * @tparam Container Type of container used to store the elements
 **/
template <typename T, typename Container = std::vector<T, std::allocator<T>>>
class ring_buffer {
  /** True if the container's non-const subscript cannot throw. */
  static constexpr bool nothrow_subscript_
      = noexcept(std::declval<Container&>()[std::declval<std::size_t>()]);
  /** True if the container's const subscript cannot throw. */
  static constexpr bool nothrow_const_subscript_
      = noexcept(std::declval<const Container&>()[std::declval<std::size_t>()]);
  /** True if querying the container's size cannot throw. */
  static constexpr bool nothrow_size_
      = noexcept(std::declval<const Container&>().size());
  /**
   * True if indexing cannot throw. Indexing consults the capacity to wrap,
   * so it is only nothrow when the container's size query is too.
   **/
  static constexpr bool nothrow_index_ = nothrow_subscript_ && nothrow_size_;
  /** True if const indexing cannot throw. */
  static constexpr bool nothrow_const_index_
      = nothrow_const_subscript_ && nothrow_size_;

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;

  /**
   * Random access iterator over the buffer's logical element order.
   *
   * @tparam Const True for the constant iterator
   **/
  template <bool Const>
  class iter_impl {
    template <bool>
    friend class iter_impl;
    friend class ring_buffer;

    using buffer_ptr
        = std::conditional_t<Const, const ring_buffer*, ring_buffer*>;
    using owner_ref
        = std::conditional_t<Const, const ring_buffer&, ring_buffer&>;

    /** True if dereferencing cannot throw, tracking the buffer's subscript. */
    static constexpr bool nothrow_deref_
        = noexcept(std::declval<owner_ref>()[std::declval<size_type>()]);

   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<Const, const T*, T*>;
    using reference = std::conditional_t<Const, const T&, T&>;

    iter_impl() = default;
    iter_impl(buffer_ptr buffer, size_type pos) noexcept
        : buffer_(buffer), pos_(pos) {}

    /** Converts a mutable iterator to a constant iterator. */
    template <bool C = Const, typename = std::enable_if_t<C>>
    iter_impl(const iter_impl<false>& other) noexcept  // NOLINT
        : buffer_(other.buffer_), pos_(other.pos_) {}

    reference operator*() const noexcept(nothrow_deref_) {
      return (*buffer_)[pos_];
    }
    pointer operator->() const noexcept(nothrow_deref_) {
      return std::addressof((*buffer_)[pos_]);
    }
    reference operator[](difference_type n) const noexcept(nothrow_deref_) {
      return (*buffer_)[pos_ + n];
    }

    iter_impl& operator++() noexcept {
      ++pos_;
      return *this;
    }
    iter_impl operator++(int) noexcept {
      iter_impl result = *this;
      ++pos_;
      return result;
    }
    iter_impl& operator--() noexcept {
      --pos_;
      return *this;
    }
    iter_impl operator--(int) noexcept {
      iter_impl result = *this;
      --pos_;
      return result;
    }

    iter_impl& operator+=(difference_type n) noexcept {
      pos_ += n;
      return *this;
    }
    iter_impl& operator-=(difference_type n) noexcept {
      pos_ -= n;
      return *this;
    }
    friend iter_impl operator+(iter_impl it, difference_type n) noexcept {
      it += n;
      return it;
    }
    friend iter_impl operator+(difference_type n, iter_impl it) noexcept {
      it += n;
      return it;
    }
    friend iter_impl operator-(iter_impl it, difference_type n) noexcept {
      it -= n;
      return it;
    }
    friend difference_type operator-(const iter_impl& a,
                                     const iter_impl& b) noexcept {
      return static_cast<difference_type>(a.pos_)
             - static_cast<difference_type>(b.pos_);
    }

    friend bool operator==(const iter_impl& a, const iter_impl& b) noexcept {
      return a.buffer_ == b.buffer_ && a.pos_ == b.pos_;
    }
    friend bool operator!=(const iter_impl& a, const iter_impl& b) noexcept {
      return !(a == b);
    }
    friend bool operator<(const iter_impl& a, const iter_impl& b) noexcept {
      return a.pos_ < b.pos_;
    }
    friend bool operator>(const iter_impl& a, const iter_impl& b) noexcept {
      return b.pos_ < a.pos_;
    }
    friend bool operator<=(const iter_impl& a, const iter_impl& b) noexcept {
      return !(b.pos_ < a.pos_);
    }
    friend bool operator>=(const iter_impl& a, const iter_impl& b) noexcept {
      return !(a.pos_ < b.pos_);
    }

   private:
    buffer_ptr buffer_ = nullptr;
    size_type pos_ = 0;
  };

  using iterator = iter_impl<false>;
  using const_iterator = iter_impl<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /**
   * Construct a buffer holding up to `capacity` elements.
   *
   * @param[in] capacity Maximum number of elements, must be positive
   * @throw std::domain_error if `capacity` is zero
   **/
  explicit ring_buffer(size_type capacity) : buf_(capacity) {
    if (capacity == 0) {
      throw std::domain_error("ring_buffer capacity must be > 0");
    }
  }

  /** Construct an empty buffer with room for a single element. */
  ring_buffer() : buf_(1) {}

  ring_buffer(const ring_buffer&) = default;
  ring_buffer& operator=(const ring_buffer&) = default;

  /**
   * Move construct, leaving the source empty and safe to reuse.
   *
   * @param[in,out] other Buffer to move from
   **/
  ring_buffer(ring_buffer&& other) noexcept(
      std::is_nothrow_move_constructible<Container>::value)
      : buf_(std::move(other.buf_)), start_(other.start_), size_(other.size_) {
    other.start_ = 0;
    other.size_ = 0;
  }

  /**
   * Move assign, leaving the source empty and safe to reuse.
   *
   * @param[in,out] other Buffer to move from
   * @return reference to this buffer
   **/
  ring_buffer& operator=(ring_buffer&& other) noexcept(
      std::is_nothrow_move_assignable<Container>::value) {
    if (this != &other) {
      buf_ = std::move(other.buf_);
      start_ = other.start_;
      size_ = other.size_;
      other.start_ = 0;
      other.size_ = 0;
    }
    return *this;
  }

  inline size_type size() const noexcept { return size_; }
  inline size_type capacity() const noexcept(nothrow_size_) {
    return buf_.size();
  }
  inline bool empty() const noexcept { return size_ == 0; }

  inline void clear() noexcept {
    start_ = 0;
    size_ = 0;
  }

  /** Advance the buffer by one element without assigning to it. */
  inline void push_back() noexcept(nothrow_size_) {
    const size_type cap = capacity();
    if (size_ < cap) {
      ++size_;
    } else if (cap > 0) {
      start_ = (start_ + 1 == cap) ? 0 : start_ + 1;
    }
  }

  /**
   * Append a value, overwriting the oldest element when full.
   *
   * @tparam U Type of the value, deduced
   * @param[in] value Value to append
   **/
  template <typename U>
  void push_back(U&& value) {
    push_back();
    back() = std::forward<U>(value);
  }

  reference front() & noexcept(nothrow_index_) { return (*this)[0]; }
  const_reference front() const& noexcept(nothrow_const_index_) {
    return (*this)[0];
  }
  T&& front() && noexcept(nothrow_index_) { return std::move((*this)[0]); }

  reference back() & noexcept(nothrow_index_) { return (*this)[size_ - 1]; }
  const_reference back() const& noexcept(nothrow_const_index_) {
    return (*this)[size_ - 1];
  }
  T&& back() && noexcept(nothrow_index_) {
    return std::move((*this)[size_ - 1]);
  }

  reference operator[](size_type i) & noexcept(nothrow_index_) {
    return buf_[offset(i)];
  }
  const_reference operator[](size_type i) const& noexcept(
      nothrow_const_index_) {
    return buf_[offset(i)];
  }
  T&& operator[](size_type i) && noexcept(nothrow_index_) {
    return std::move(buf_[offset(i)]);
  }

  /**
   * Return the element at index `i` with bounds checking.
   *
   * @param[in] i Index into the buffer's logical order
   * @return reference to the element
   * @throw std::out_of_range if `i` is not less than `size()`
   **/
  reference at(size_type i) & {
    check_index(i);
    return buf_[offset(i)];
  }
  const_reference at(size_type i) const& {
    check_index(i);
    return buf_[offset(i)];
  }

  iterator begin() noexcept { return iterator(this, 0); }
  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator cbegin() const noexcept { return const_iterator(this, 0); }

  iterator end() noexcept { return iterator(this, size_); }
  const_iterator end() const noexcept { return const_iterator(this, size_); }
  const_iterator cend() const noexcept { return const_iterator(this, size_); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }

  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

  /**
   * Exchange contents with another buffer.
   *
   * @param[in,out] other Buffer to swap with
   **/
  void swap(ring_buffer& other) noexcept(
      std::is_nothrow_swappable<Container>::value) {
    using std::swap;
    swap(buf_, other.buf_);
    swap(start_, other.start_);
    swap(size_, other.size_);
  }

  friend void swap(ring_buffer& a,
                   ring_buffer& b) noexcept(noexcept(a.swap(b))) {
    a.swap(b);
  }

  friend bool operator==(const ring_buffer& a, const ring_buffer& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
  }
  friend bool operator!=(const ring_buffer& a, const ring_buffer& b) {
    return !(a == b);
  }
  friend bool operator<(const ring_buffer& a, const ring_buffer& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  }
  friend bool operator>(const ring_buffer& a, const ring_buffer& b) {
    return b < a;
  }
  friend bool operator<=(const ring_buffer& a, const ring_buffer& b) {
    return !(b < a);
  }
  friend bool operator>=(const ring_buffer& a, const ring_buffer& b) {
    return !(a < b);
  }

  /**
   * Resize the buffer, keeping the most recently added elements.
   *
   * @param[in] new_capacity Maximum number of elements, must be positive
   * @throw std::domain_error if `new_capacity` is zero
   **/
  void reset_capacity(size_type new_capacity) {
    if (new_capacity == 0) {
      throw std::domain_error("ring_buffer capacity must be > 0");
    }
    if (new_capacity == capacity()) {
      return;
    }

    Container new_buf(new_capacity);
    size_type keep = std::min(size_, new_capacity);
    for (size_type i = 0; i < keep; ++i) {
      new_buf[i] = std::move((*this)[size_ - keep + i]);
    }
    buf_ = std::move(new_buf);
    start_ = 0;
    size_ = keep;
  }

 private:
  /**
   * Map a logical index onto the backing container's index.
   *
   * A conditional subtraction replaces a modulo, which keeps integer
   * division out of the hot path and leaves a zero-capacity buffer with
   * ordinary out-of-contract behavior rather than a division by zero.
   *
   * @param[in] i Index into the buffer's logical order
   * @return index into the backing container
   **/
  inline size_type offset(size_type i) const noexcept(nothrow_size_) {
    const size_type cap = capacity();
    const size_type j = start_ + i;
    return j >= cap ? j - cap : j;
  }

  inline void check_index(size_type i) const {
    if (i >= size_) {
      throw std::out_of_range("ring_buffer index out of range");
    }
  }

  Container buf_;
  size_type start_ = 0;
  size_type size_ = 0;
};

}  // namespace util
}  // namespace stan
#endif
