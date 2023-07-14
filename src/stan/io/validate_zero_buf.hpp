#ifndef STAN_IO_VALIDATE_ZERO_BUF_HPP
#define STAN_IO_VALIDATE_ZERO_BUF_HPP

#include <stdexcept>
#include <string>

namespace stan {
namespace io {

/**
 * Throw an bad-cast exception if the specified buffer contains
 * a digit other than 0 before an e or E.  The buffer argument
 * must implement <code>size_t size()</code> method and <code>char
 * operator[](size_t)</code>.
 *
 * @tparam B Character buffer type
 * @throw <code>std::invalid_argument</code> if the buffer
 * contains non-zero digit before an exponentiation symbol.
 */
template <typename B>
void validate_zero_buf(const B& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] == 'e' || buf[i] == 'E')
      return;
    if (buf[i] >= '1' && buf[i] <= '9')
      throw std::invalid_argument("non-zero digit before E");
  }
}

}  // namespace io
}  // namespace stan
#endif
