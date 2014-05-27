#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_DIMS_HPP__

#include <sstream>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1, int C1, int R2, int C2,
              typename T_result>
    inline bool check_matching_dims(const char* function,
                                    const Eigen::Matrix<T1,R1,C1>& y1,
                                    const char* name1,
                                    const Eigen::Matrix<T2,R2,C2>& y2,
                                    const char* name2,
                                    T_result* result) {
      if ((y1.rows() == y2.rows()) && (y1.cols() == y2.cols()))
        return true;

      std::ostringstream msg;
      msg << name1 << " (%1%) has " << y1.rows() << " rows and " 
          << y1.cols() << " columns and " << name2 << " (" << y2 
          << ") has " << y2.rows() << " rows and " << y2.cols()
          << " columns but they must match in dimensions";
      std::string tmp(msg.str());
      return dom_err(function,y1,name1,
                     tmp.c_str(),"",
                     result);
    }

  }
}
#endif
