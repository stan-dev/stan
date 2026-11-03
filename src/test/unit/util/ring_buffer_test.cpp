#include <gtest/gtest.h>
#include <stan/util/ring_buffer.hpp>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using stan::util::ring_buffer;

namespace {

/**
 * Element type that records how many times it has been moved, so tests can
 * prove that the rvalue-qualified accessors actually move rather than copy.
 */
struct tracked {
  int value = 0;
  int moves = 0;
  tracked() = default;
  explicit tracked(int v) : value(v) {}
  tracked(const tracked&) = default;
  tracked& operator=(const tracked&) = default;
  tracked(tracked&& other) noexcept
      : value(other.value), moves(other.moves + 1) {
    other.value = -1;
  }
  tracked& operator=(tracked&& other) noexcept {
    value = other.value;
    moves = other.moves + 1;
    other.value = -1;
    return *this;
  }
};

/**
 * Backing store whose subscript operators are *not* noexcept, used to check
 * that ring_buffer's exception specifications track the container's.
 */
template <typename T>
struct checked_vector : std::vector<T> {
  using std::vector<T>::vector;
  T& operator[](std::size_t i) { return this->at(i); }
  const T& operator[](std::size_t i) const { return this->at(i); }
};

/**
 * Backing store whose size() may throw, used to check that accessors which
 * consult the capacity report that in their exception specification.
 */
template <typename T>
struct loud_size_vector : std::vector<T> {
  using std::vector<T>::vector;
  // deliberately not noexcept, unlike std::vector::size
  std::size_t size() const { return std::vector<T>::size(); }
};

struct point {
  int x = 0;
  int y = 0;
};

ring_buffer<int> filled(std::size_t capacity, std::size_t n) {
  ring_buffer<int> b(capacity);
  for (std::size_t i = 0; i < n; ++i) {
    b.push_back(static_cast<int>(i));
  }
  return b;
}

std::vector<int> to_vector(const ring_buffer<int>& b) {
  return std::vector<int>(b.begin(), b.end());
}

}  // namespace

// ---------------------------------------------------------------- typedefs

TEST(RingBuffer, exposes_standard_member_typedefs) {
  using rb = ring_buffer<int>;
  static_assert(std::is_same<rb::value_type, int>::value, "value_type");
  static_assert(std::is_same<rb::size_type, std::size_t>::value, "size_type");
  static_assert(std::is_same<rb::difference_type, std::ptrdiff_t>::value,
                "difference_type");
  static_assert(std::is_same<rb::reference, int&>::value, "reference");
  static_assert(std::is_same<rb::const_reference, const int&>::value,
                "const_reference");
  static_assert(std::is_same<rb::pointer, int*>::value, "pointer");
  static_assert(std::is_same<rb::const_pointer, const int*>::value,
                "const_pointer");
  SUCCEED();
}

// --------------------------------------------------------------- iterators

TEST(RingBuffer, iterators_are_random_access) {
  using rb = ring_buffer<int>;
  static_assert(
      std::is_same<std::iterator_traits<rb::iterator>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "iterator must be random access");
  static_assert(
      std::is_same<std::iterator_traits<rb::const_iterator>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "const_iterator must be random access");
  SUCCEED();
}

TEST(RingBuffer, iterator_converts_to_const_iterator_but_not_back) {
  using rb = ring_buffer<int>;
  static_assert(std::is_convertible<rb::iterator, rb::const_iterator>::value,
                "iterator -> const_iterator");
  static_assert(!std::is_convertible<rb::const_iterator, rb::iterator>::value,
                "const_iterator must not convert to iterator");
  SUCCEED();
}

TEST(RingBuffer, begin_on_const_buffer_yields_const_iterator) {
  using rb = ring_buffer<int>;
  static_assert(
      std::is_same<decltype(std::declval<rb&>().begin()), rb::iterator>::value,
      "non-const begin");
  static_assert(std::is_same<decltype(std::declval<const rb&>().begin()),
                             rb::const_iterator>::value,
                "const begin");
  static_assert(std::is_same<decltype(std::declval<rb&>().cbegin()),
                             rb::const_iterator>::value,
                "cbegin");
  static_assert(std::is_same<decltype(std::declval<rb&>().crbegin()),
                             rb::const_reverse_iterator>::value,
                "crbegin");
  SUCCEED();
}

TEST(RingBuffer, mutable_iterator_can_modify_elements) {
  ring_buffer<int> b = filled(4, 4);
  for (auto it = b.begin(); it != b.end(); ++it) {
    *it *= 10;
  }
  EXPECT_EQ(std::vector<int>({0, 10, 20, 30}), to_vector(b));
}

TEST(RingBuffer, iterator_supports_random_access_arithmetic) {
  ring_buffer<int> b = filled(5, 5);
  auto first = b.begin();
  auto last = b.end();
  EXPECT_EQ(5, last - first);
  EXPECT_EQ(5, std::distance(first, last));
  EXPECT_EQ(2, *(first + 2));
  EXPECT_EQ(2, first[2]);
  EXPECT_EQ(4, *(last - 1));
  EXPECT_TRUE(first < last);
  EXPECT_TRUE(last > first);
  EXPECT_TRUE(first <= first);
  EXPECT_TRUE(last >= last);
  auto mid = first;
  mid += 3;
  EXPECT_EQ(3, *mid);
  mid -= 2;
  EXPECT_EQ(1, *mid);
}

TEST(RingBuffer, iterator_supports_arrow) {
  ring_buffer<point> b(2);
  b.push_back(point{1, 2});
  EXPECT_EQ(1, b.begin()->x);
  EXPECT_EQ(2, b.begin()->y);
}

TEST(RingBuffer, iteration_follows_logical_order_after_wrapping) {
  ring_buffer<int> b = filled(3, 5);  // 0,1 overwritten
  EXPECT_EQ(3u, b.size());
  EXPECT_EQ(std::vector<int>({2, 3, 4}), to_vector(b));
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(4, b[2]);
}

TEST(RingBuffer, reverse_iteration_visits_newest_first) {
  ring_buffer<int> b = filled(3, 5);
  std::vector<int> seen(b.rbegin(), b.rend());
  EXPECT_EQ(std::vector<int>({4, 3, 2}), seen);
  std::vector<int> cseen(b.crbegin(), b.crend());
  EXPECT_EQ(std::vector<int>({4, 3, 2}), cseen);
}

TEST(RingBuffer, works_with_standard_algorithms) {
  ring_buffer<int> b = filled(4, 4);
  EXPECT_EQ(6, std::accumulate(b.begin(), b.end(), 0));
  EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));
}

// --------------------------------------------------------- element access

TEST(RingBuffer, element_access_is_reference_qualified) {
  using rb = ring_buffer<int>;
  static_assert(std::is_same<decltype(std::declval<rb&>()[0]), int&>::value,
                "lvalue subscript");
  static_assert(
      std::is_same<decltype(std::declval<const rb&>()[0]), const int&>::value,
      "const lvalue subscript");
  static_assert(std::is_same<decltype(std::declval<rb&&>()[0]), int&&>::value,
                "rvalue subscript");
  static_assert(std::is_same<decltype(std::declval<rb&>().back()), int&>::value,
                "lvalue back");
  static_assert(std::is_same<decltype(std::declval<const rb&>().back()),
                             const int&>::value,
                "const back");
  static_assert(
      std::is_same<decltype(std::declval<rb&&>().back()), int&&>::value,
      "rvalue back");
  static_assert(
      std::is_same<decltype(std::declval<rb&&>().front()), int&&>::value,
      "rvalue front");
  SUCCEED();
}

TEST(RingBuffer, rvalue_subscript_moves_out_of_the_element) {
  ring_buffer<tracked> b(2);
  b.push_back(tracked(7));
  const int moves_before = b[0].moves;
  tracked taken = std::move(b)[0];
  EXPECT_EQ(7, taken.value);
  EXPECT_GT(taken.moves, moves_before);
}

TEST(RingBuffer, rvalue_back_moves_out_of_the_element) {
  ring_buffer<tracked> b(2);
  b.push_back(tracked(9));
  tracked taken = std::move(b).back();
  EXPECT_EQ(9, taken.value);
  EXPECT_GT(taken.moves, 0);
}

TEST(RingBuffer, front_and_back_track_the_logical_ends) {
  ring_buffer<int> b = filled(3, 5);
  EXPECT_EQ(2, b.front());
  EXPECT_EQ(4, b.back());
  const ring_buffer<int>& cb = b;
  EXPECT_EQ(2, cb.front());
  EXPECT_EQ(4, cb.back());
}

TEST(RingBuffer, empty_reports_logical_emptiness) {
  ring_buffer<int> b(4);
  EXPECT_TRUE(b.empty());
  b.push_back(1);
  EXPECT_FALSE(b.empty());
  b.clear();
  EXPECT_TRUE(b.empty());
}

TEST(RingBuffer, at_throws_out_of_range_past_the_end) {
  ring_buffer<int> b = filled(4, 2);
  EXPECT_EQ(1, b.at(1));
  EXPECT_THROW(b.at(2), std::out_of_range);
  const ring_buffer<int>& cb = b;
  EXPECT_THROW(cb.at(2), std::out_of_range);
}

// -------------------------------------------------------- move semantics

TEST(RingBuffer, move_construction_leaves_source_empty_and_usable) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b(std::move(a));
  EXPECT_EQ(std::vector<int>({0, 1, 2}), to_vector(b));
  EXPECT_EQ(0u, a.size());
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(a.begin(), a.end());
  EXPECT_EQ(0, std::distance(a.begin(), a.end()));
}

TEST(RingBuffer, move_assignment_leaves_source_empty_and_usable) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b(2);
  b = std::move(a);
  EXPECT_EQ(std::vector<int>({0, 1, 2}), to_vector(b));
  EXPECT_EQ(0u, a.size());
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(a.begin(), a.end());
}

TEST(RingBuffer, moved_from_buffer_can_be_refilled) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b(std::move(a));
  a.reset_capacity(2);
  a.push_back(42);
  EXPECT_EQ(1u, a.size());
  EXPECT_EQ(42, a.back());
}

TEST(RingBuffer, move_operations_are_noexcept) {
  using rb = ring_buffer<int>;
  static_assert(std::is_nothrow_move_constructible<rb>::value,
                "move ctor noexcept");
  static_assert(std::is_nothrow_move_assignable<rb>::value,
                "move assign noexcept");
  SUCCEED();
}

TEST(RingBuffer, copy_construction_is_independent) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b(a);
  b[0] = 99;
  EXPECT_EQ(0, a[0]);
  EXPECT_EQ(99, b[0]);
  EXPECT_EQ(3u, a.size());
}

// ------------------------------------------------------ swap / comparison

TEST(RingBuffer, swap_exchanges_contents) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b = filled(2, 2);
  a.swap(b);
  EXPECT_EQ(std::vector<int>({0, 1}), to_vector(a));
  EXPECT_EQ(std::vector<int>({0, 1, 2}), to_vector(b));
}

TEST(RingBuffer, free_swap_is_found_by_adl) {
  ring_buffer<int> a = filled(4, 3);
  ring_buffer<int> b = filled(2, 2);
  using std::swap;
  swap(a, b);
  EXPECT_EQ(2u, a.size());
  EXPECT_EQ(3u, b.size());
}

TEST(RingBuffer, equality_compares_logical_contents_not_capacity) {
  ring_buffer<int> a(3);
  ring_buffer<int> b(8);
  a.push_back(1);
  a.push_back(2);
  b.push_back(1);
  b.push_back(2);
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  b.push_back(3);
  EXPECT_TRUE(a != b);
}

TEST(RingBuffer, equality_ignores_overwritten_elements) {
  ring_buffer<int> a = filled(2, 5);  // holds 3,4
  ring_buffer<int> b(2);
  b.push_back(3);
  b.push_back(4);
  EXPECT_TRUE(a == b);
}

TEST(RingBuffer, relational_operators_are_lexicographical) {
  ring_buffer<int> a(4);
  ring_buffer<int> b(4);
  a.push_back(1);
  a.push_back(2);
  b.push_back(1);
  b.push_back(3);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(b < a);
}

// ----------------------------------------------------- conditional noexcept

TEST(RingBuffer, subscript_noexcept_tracks_the_container) {
  // Asserted as a *relationship*, not a hardcoded true: whether
  // std::vector::operator[] is noexcept is implementation defined.
  using vec_rb = ring_buffer<int>;
  constexpr bool vec_ok
      = noexcept(std::declval<std::vector<int>&>()[std::size_t{0}]);
  static_assert(noexcept(std::declval<vec_rb&>()[0]) == vec_ok,
                "vector-backed subscript must match the container");
  static_assert(noexcept(*std::declval<vec_rb::iterator&>()) == vec_ok,
                "iterator deref must match the container");

  using checked_rb = ring_buffer<int, checked_vector<int>>;
  static_assert(!noexcept(std::declval<checked_rb&>()[0]),
                "throwing container must make subscript throwing");
  static_assert(!noexcept(*std::declval<checked_rb::iterator&>()),
                "throwing container must make iterator deref throwing");
  static_assert(!noexcept(std::declval<checked_rb&>().back()),
                "throwing container must make back() throwing");
  SUCCEED();
}

TEST(RingBuffer, iterator_navigation_is_unconditionally_noexcept) {
  using checked_rb = ring_buffer<int, checked_vector<int>>;
  static_assert(noexcept(++std::declval<checked_rb::iterator&>()),
                "increment never touches the container");
  static_assert(noexcept(std::declval<checked_rb::iterator&>()
                         == std::declval<checked_rb::iterator&>()),
                "comparison never touches the container");
  static_assert(noexcept(std::declval<checked_rb&>().begin()),
                "begin never touches the container");
  SUCCEED();
}

TEST(RingBuffer, subscript_noexcept_accounts_for_the_container_size_call) {
  // Indexing consults capacity() to wrap, so a container whose size() can
  // throw must not yield a noexcept subscript.
  using rb = ring_buffer<int, loud_size_vector<int>>;
  static_assert(!noexcept(std::declval<rb&>().capacity()),
                "capacity must follow the container's size()");
  static_assert(!noexcept(std::declval<rb&>()[0]),
                "subscript consults capacity, so it cannot claim noexcept");
  static_assert(!noexcept(std::declval<const rb&>()[0]),
                "const subscript consults capacity");
  static_assert(!noexcept(std::declval<rb&>().back()),
                "back consults capacity");
  static_assert(!noexcept(*std::declval<rb::iterator&>()),
                "iterator deref consults capacity");
  SUCCEED();
}

// ------------------------------------------------------------- capacity

TEST(RingBuffer, default_constructed_buffer_has_capacity_one) {
  ring_buffer<int> b;
  EXPECT_EQ(1u, b.capacity());
  EXPECT_EQ(0u, b.size());
  EXPECT_TRUE(b.empty());
  b.push_back(5);
  EXPECT_EQ(1u, b.size());
  EXPECT_EQ(5, b.back());
}

TEST(RingBuffer, constructing_with_zero_capacity_throws) {
  EXPECT_THROW(ring_buffer<int>(0), std::domain_error);
}

TEST(RingBuffer, reset_capacity_to_zero_throws) {
  ring_buffer<int> b(4);
  EXPECT_THROW(b.reset_capacity(0), std::domain_error);
}

TEST(RingBuffer, reset_capacity_shrink_keeps_newest_elements) {
  ring_buffer<int> b = filled(5, 5);
  b.reset_capacity(2);
  EXPECT_EQ(2u, b.capacity());
  EXPECT_EQ(std::vector<int>({3, 4}), to_vector(b));
}

TEST(RingBuffer, reset_capacity_grow_keeps_all_elements) {
  ring_buffer<int> b = filled(3, 5);
  b.reset_capacity(6);
  EXPECT_EQ(6u, b.capacity());
  EXPECT_EQ(std::vector<int>({2, 3, 4}), to_vector(b));
  b.push_back(5);
  EXPECT_EQ(std::vector<int>({2, 3, 4, 5}), to_vector(b));
}

TEST(RingBuffer, reset_capacity_uses_the_container_type) {
  // Regression: reset_capacity used to build a std::vector<T> unconditionally,
  // which fails to compile for any other backing store.
  ring_buffer<int, checked_vector<int>> b(3);
  b.push_back(1);
  b.push_back(2);
  b.reset_capacity(4);
  EXPECT_EQ(4u, b.capacity());
  EXPECT_EQ(2u, b.size());
  EXPECT_EQ(1, b[0]);
  EXPECT_EQ(2, b[1]);
}

TEST(RingBuffer, clear_resets_size_but_not_capacity) {
  ring_buffer<int> b = filled(4, 4);
  b.clear();
  EXPECT_EQ(0u, b.size());
  EXPECT_EQ(4u, b.capacity());
  EXPECT_EQ(b.begin(), b.end());
}
