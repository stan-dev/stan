#ifndef __STAN__MATH__BOOST_ERROR_HANDLING_HPP__
#define __STAN__MATH__BOOST_ERROR_HANDLING_HPP__

#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

namespace stan {
  namespace math {
    namespace policies {
      namespace detail {
        using boost::math::policies::detail::raise_error;
        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* function, 
                                           const char* message, 
                                           const T_val& val, 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::throw_on_error>&) {
          raise_error<std::domain_error, T_val>(function, message, val);
          // we never get here:
          return std::numeric_limits<T_result>::quiet_NaN();
        }

        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* , 
                                           const char* , 
                                           const T_val& , 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::ignore_error>&) {
          // This may or may not do the right thing, but the user asked for the error
          // to be ignored so here we go anyway:
          return std::numeric_limits<T_result>::quiet_NaN();
        }

        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* , 
                                           const char* , 
                                           const T_val& , 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::errno_on_error>&) {
          errno = EDOM;
          // This may or may not do the right thing, but the user asked for the error
          // to be silent so here we go anyway:
          return std::numeric_limits<T_result>::quiet_NaN();
        }
       
        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* function, 
                                           const char* message, 
                                           const T_val& val, 
                                           const  ::boost::math::policies::domain_error< ::boost::math::policies::user_error>&) {
          return user_domain_error(function, message, val);
        }

      }
    
      template <class T_result, class T_val, class Policy>
      inline T_result raise_domain_error(const char* function, 
                          const char* message, 
                          const T_val& val, 
                          const Policy&) {
        typedef typename Policy::domain_error_type policy_type;
        return detail::raise_domain_error<T_result,T_val>(function, 
                                                          message ? message : "Domain Error evaluating function at %1%", 
                                                          val, 
                                                          policy_type());
      
      }
    }
  }
  
}

// FIXME: remove when BOOST fixes isfinite(). See ticket #6517. (Boost 1.48.0)
//    https://svn.boost.org/trac/boost/ticket/6517
namespace boost {

  namespace math {   

    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(short n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(short n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(short n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(short n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned short n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(unsigned short n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(unsigned short n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned short n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(int n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(int n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(int n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(int n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned int n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(unsigned int n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(unsigned int n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned int n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(long n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(long n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(long n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(long n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned long n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(unsigned long n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(unsigned long n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned long n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(long long n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(long long n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(long long n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(long long n) {
      return true;
    }


    /**
     * Checks if the given number has finite value.
     *
     * Integer values are always finite.
     * Returns <code>true</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned long long n) {
      return true;
    }
    /** 
     * Checks if the given number is infinite.
     * 
     * Integer values are never infinite. 
     * Returns <code>false</code> for all arguments.
     * 
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isinf(unsigned long long n) {
      return false;
    }
    /** 
     * Checks if the given number is NaN
     * 
     * Integer values are never NaN.
     * Returns <code>false</code> for all arguments.
     * @param n Value to test.
     * @return <code>false</code>
     */
    template <>
    inline
    bool isnan(unsigned long long n) {
      return false;
    }
    /** 
     * Checks if the given number is normal.
     * 
     * Integer values are always normal.
     * Return <code>true</code> for all arguments.
     *
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned long long n) {
      return true;
    }

  }

}

#endif
