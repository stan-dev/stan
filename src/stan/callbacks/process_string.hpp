#ifndef STAN_CALLBACKS_PROCESS_STRING_HPP
#define STAN_CALLBACKS_PROCESS_STRING_HPP

#include <string>

namespace stan {
namespace callbacks {

/**
 * Process a string to escape the following special characters:
 * `'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'`.
 *
 * @param value The string to process.
 * @return The processed string.
 */
std::string process_string(const std::string& value) {
  static constexpr std::array<char, 11> chars_to_escape
      = {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'};
  static constexpr std::array<const char*, 11> chars_to_replace
      = {"\\\\", "\\\"", "\\/", "\\b", "\\f", "\\n",
         "\\r",  "\\t",  "\\v", "\\a", "\\0"};
  // Replacing every value leads to 2x the size
  std::string new_value(value.size() * 2, 'x');
  std::size_t pos = 0;
  std::size_t count = 0;
  std::size_t prev_pos = 0;
  while ((pos = value.find_first_of(chars_to_escape.data(), pos, 10))
         != std::string::npos) {
    for (int i = prev_pos; i < pos; ++i) {
      new_value[i + count] = value[i];
    }
    int idx
        = strchr(chars_to_escape.data(), value[pos]) - chars_to_escape.data();
    new_value[pos + count] = chars_to_replace[idx][0];
    new_value[pos + count + 1] = chars_to_replace[idx][1];
    pos += 1;
    count++;
    prev_pos = pos;
  }
  for (int i = prev_pos; i < value.size(); ++i) {
    new_value[i + count] = value[i];
  }
  // Shrink any unused space
  new_value.resize(value.size() + count);
  return new_value;
}

}  // namespace callbacks
}  // namespace stan
#endif
