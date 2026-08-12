#ifndef STAN_IO_STRING_UTILS_HPP
#define STAN_IO_STRING_UTILS_HPP

#include <algorithm>
#include <cctype>
#include <string>
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
 * @param s string to split
 * @param delims set of delimiter characters
 * @param compress_delims whether to collapse adjacent delimiters
 * @return the tokens found in `s`
 */
inline std::vector<std::string> split(const std::string& s,
                                      const std::string& delims,
                                      bool compress_delims = false) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (true) {
    size_t pos = s.find_first_of(delims, start);
    if (pos == std::string::npos) {
      tokens.push_back(s.substr(start));
      break;
    }
    tokens.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  if (compress_delims) {
    std::vector<std::string> compressed;
    for (size_t i = 0; i < tokens.size(); ++i) {
      // drop empty tokens produced by interior runs of delimiters,
      // keeping a single empty token at either end
      if (tokens[i].empty() && i > 0 && i + 1 < tokens.size())
        continue;
      compressed.push_back(std::move(tokens[i]));
    }
    tokens.swap(compressed);
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
                        const std::string& sep) {
  std::string result;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0)
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
inline void replace_first(std::string& s, const std::string& target,
                          const std::string& replacement) {
  size_t pos = s.find(target);
  if (pos != std::string::npos)
    s.replace(pos, target.size(), replacement);
}

}  // namespace io
}  // namespace stan
#endif
