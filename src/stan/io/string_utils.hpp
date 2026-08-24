#ifndef STAN_IO_STRING_UTILS_HPP
#define STAN_IO_STRING_UTILS_HPP

#include <string>
#include <string_view>
#include <vector>

namespace stan {
namespace io {

/**
 * Joins a vector of strings/string_views into a single string,
 * separated by `delimiter`.
 *
 * @param container strings to join
 * @param delimiter separator inserted between parts
 * @return the joined string
 */
template <typename T>
inline std::string join(T&& container, const std::string_view& delimiter) {
  if (container.empty()) {
    return "";
  }
  std::size_t total = (container.size() - 1) * delimiter.size();
  for (const auto& element : container) {
    total += std::string_view(element).size();
  }
  std::string result;
  result.reserve(total);
  bool first = true;
  for (const auto& element : container) {
    if (!first) {
      result.append(delimiter);
    }
    result.append(std::string_view(element));
    first = false;
  }
  return result;
}

/**
 * Splits a string on any character found in `delimiters`.
 *
 * Empty substrings are preserved. Empty delimiters return the input as a
 * single element. When `compress_delims` is true, consecutive delimiters are
 * treated as a single delimiter.
 *
 * @param input string to split
 * @param delimiters separators between parts
 * @param compress_tokens whether to combine consecutive delimiters
 * @return the split strings
 */
inline std::vector<std::string> split(const std::string_view& input,
                                      const std::string_view& delimiters,
                                      bool compress_delims = false) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= input.size()) {
    const std::size_t end = input.find_first_of(delimiters, start);
    if (end == std::string_view::npos) {
      result.emplace_back(input.substr(start));
      break;
    }
    result.emplace_back(input.substr(start, end - start));
    start = end + 1;
    if (compress_delims) {
      const std::size_t next = input.find_first_not_of(delimiters, start);
      start = (next == std::string_view::npos) ? input.size() : next;
    }
  }
  return result;
}

/**
 * Remove leading and trailing whitespace from a string in place.
 *
 * @param input string to trim
 */
inline void trim(std::string& input) {
  constexpr std::string_view whitespace = " \t\n\r\f\v";
  const std::size_t start = input.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    input.clear();
    return;
  }
  const std::size_t end = input.find_last_not_of(whitespace);
  input.erase(end + 1);
  input.erase(0, start);
}

/**
 * Removes the first occurrence of a substring from a string in place.
 *
 * @param input string to modify
 * @param substring substring to remove
 */
inline void remove_first(std::string& input,
                         const std::string_view& substring) {
  const std::size_t position = input.find(substring);
  if (position != std::string::npos) {
    input.erase(position, substring.size());
  }
}

}
}

#endif
