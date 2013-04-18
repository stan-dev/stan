#ifndef __STAN__MATH__MATRIX__VALIDATE_MATCHING_SIZES_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_MATCHING_SIZES_HPP__

#include <sstream>
#include <stdexcept>
#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2>
    inline void validate_matching_sizes(const std::vector<T1>& x1,
                                        const std::vector<T2>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching sizes, but found"
         << " arg1(size=" << x1.size() << ");"
         << " arg2(size=" << x2.size() << ");";
      throw std::domain_error(ss.str());
    }

    template <typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline void validate_matching_sizes(const Eigen::Matrix<T1,R1,C1>& x1,
                                        const Eigen::Matrix<T2,R2,C2>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching sizes, but found"
         << " arg1(rows=" << x1.rows() << ",cols=" << x1.cols() 
         << ",size=" << (x1.rows() * x1.cols()) << ");"
         << " arg2(rows=" << x2.rows() << ",cols=" << x2.cols() 
         << ",size=" << (x2.rows() * x2.cols()) << ")";
      throw std::domain_error(ss.str());
    }

    template <typename Derived, typename T2, int R2, int C2>
    inline void validate_matching_sizes(const Eigen::Block<Derived>& x1,
                                        const Eigen::Matrix<T2,R2,C2>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching sizes, but found"
         << " arg1(rows=" << x1.rows() << ",cols=" << x1.cols() 
         << ",size=" << (x1.rows() * x1.cols()) << ");"
         << " arg2(rows=" << x2.rows() << ",cols=" << x2.cols() 
         << ",size=" << (x2.rows() * x2.cols()) << ")";
      throw std::domain_error(ss.str());
    }

    template <typename T1, int R1, int C1, typename Derived>
    inline void validate_matching_sizes(const Eigen::Matrix<T1,R1,C1>& x1,
                                        const Eigen::Block<Derived>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching sizes, but found"
         << " arg1(rows=" << x1.rows() << ",cols=" << x1.cols() 
         << ",size=" << (x1.rows() * x1.cols()) << ");"
         << " arg2(rows=" << x2.rows() << ",cols=" << x2.cols() 
         << ",size=" << (x2.rows() * x2.cols()) << ")";
      throw std::domain_error(ss.str());
    }

    template <typename Derived1, typename Derived2>
    inline void validate_matching_sizes(const Eigen::Block<Derived1>& x1,
                                        const Eigen::Block<Derived2>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching sizes, but found"
         << " arg1(rows=" << x1.rows() << ",cols=" << x1.cols() 
         << ",size=" << (x1.rows() * x1.cols()) << ");"
         << " arg2(rows=" << x2.rows() << ",cols=" << x2.cols() 
         << ",size=" << (x2.rows() * x2.cols()) << ")";
      throw std::domain_error(ss.str());
    }
    
  }
}
#endif
