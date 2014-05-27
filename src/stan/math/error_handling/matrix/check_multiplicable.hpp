#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP__

#include <sstream>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, typename T_result>
    inline bool check_multiplicable(const char* function,
                                    const T1& y1,
                                    const char* name1,
                                    const T2& y2,
                                    const char* name2,
                                    T_result* result) {
      if (y1.cols() == static_cast<typename T1::size_type>(y2.rows()))    
        return true;

      std::ostringstream msg;
      msg << " (" << typeid(T1).name() << ") has %1% columns and "
          << " (" << typeid(T2).name() << ") has " << y2.rows()
          << " rows but the number of columns in the first argument must equal the number of rows in the second argument";
      std::string tmp(msg.str());
      return dom_err(function,typename T1::value_type(),name1,
                     tmp.c_str(),"",
                     result);
    }

  }
}
#endif
