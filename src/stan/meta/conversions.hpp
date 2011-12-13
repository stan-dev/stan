#ifndef __STAN__META__CONVERSIONS_HPP__
#define __STAN__META__CONVERSIONS_HPP__

namespace stan {

  /**
   * Convert the specified object to a double.
   *
   * @param x Object to convert.
   * @return Object converted to double.
   * @tparam Type of object converted.
   */
  template <typename T>
  inline double convert(const T x) {
    return x;
  };


}

#endif
