#ifndef STAN_SRC_TEST_UNIT_SERVICES_UTIL_HPP
#define STAN_SRC_TEST_UNIT_SERVICES_UTIL_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <test/unit/util.hpp>
#include <iostream>
#include <string>

namespace stan {
namespace test {
/**
 * Read a CSV into an Eigen matrix.
 * @param in An input string stream holding the CSV
 * @param rows Number of rows
 * @param cols Number of columns.
 */
Eigen::MatrixXd read_stan_sample_csv(std::istringstream& in, int rows,
                                     int cols) {
  std::string line;
  int row = 0;
  int col = 0;
  Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);
  while (std::getline(in, line)) {
    if (line.find("#") != std::string::npos) {
      continue;
    }
    const char* ptr = line.c_str();
    int len = line.length();
    col = 0;

    const char* start = ptr;
    for (int i = 0; i < len; i++) {
      if (ptr[i] == ',') {
        res(row, col++) = atof(start);
        start = ptr + i + 1;
      }
    }
    res(row, col) = atof(start);
    row++;
  }
  return res;
}
}  // namespace test
}  // namespace stan
#endif
