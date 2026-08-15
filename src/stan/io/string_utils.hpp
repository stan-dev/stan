#ifndef STAN_IO_STRING_UTILS_HPP
#define STAN_IO_STRING_UTILS_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace stan {
namespace io {

/**
 * Splits a string on any character found in `delims`.
 *
 * If `compress_delims` is false, every delimiter produces a split, so
 * adjacent delimiters yield an empty token between them (e.g. splitting
 * "a,,b" on "," gives {"a", "", "b"}).
 *
 * If `compress_delims` is true, runs of adjacent delimiters are treated
 * as a single delimiter, except that a leading or trailing run still
 * yields a single empty token at that end (matching the behavior of
 * boost::split with token_compress_on).
 *
 * @param s sequence to split
 * @param delims set of delimiter characters
 * @param compress_delims whether to collapse adjacent delimiters
 * @return the tokens found in `s`
 */
template <typename T>
inline std::vector<std::string> split(T&& s, std::string_view delims,
                                      bool compress_delims = false) {
  constexpr std::string_view::size_type npos = std::string_view::npos;
  const std::string_view sv{s};
  std::vector<std::string> tokens;
  std::size_t start = 0;
  while (true) {
    std::size_t end = sv.find_first_of(delims, start);
    bool last = (end == npos);
    std::size_t len = last ? sv.size() - start : end - start;
    // when compressing, skip empty tokens but keep the first and last ones
    if (last || !compress_delims || len > 0 || tokens.empty()) {
      tokens.emplace_back(sv.substr(start, len));
    }
    if (last) {
      break;
    }
    start = end + 1;
    if (compress_delims) {
      // skip the rest of this run of delimiters in one search
      std::size_t next = sv.find_first_not_of(delims, start);
      start = (next == npos) ? sv.size() : next;
    }
  }
  return tokens;
}

/**
 * Joins a vector of strings into a single string, separated by `sep`.
 *
 * @param parts strings to join
 * @param sep separator inserted between parts
 * @return the joined string
 */
inline std::string join(const std::vector<std::string>& parts,
                        std::string_view sep) {
  if (parts.empty()) {
    return "";
  }
  std::size_t total = (parts.size() - 1) * sep.size();
  for (const auto& part : parts) {
    total += part.size();
  }
  std::string result;
  result.reserve(total);
  result += parts[0];
  for (std::size_t i = 1; i < parts.size(); ++i) {
    result += sep;
    result += parts[i];
  }
  return result;
}

/**
 * Trims leading and trailing whitespace from a string, in place.
 *
 * @param s string to trim
 */
inline void trim(std::string& s) {
  auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

/**
 * Replaces the first occurrence of `target` in `s` with `replacement`,
 * in place. Does nothing if `target` is not found.
 *
 * @param s string to modify
 * @param target substring to replace
 * @param replacement replacement text
 */
inline void replace_first(std::string& s, std::string_view target,
                          std::string_view replacement) {
  std::size_t pos = s.find(target);
  if (pos != std::string::npos) {
    s.replace(pos, target.size(), replacement.data(), replacement.size());
  }
}

}  // namespace io
}  // namespace stan
#endif
