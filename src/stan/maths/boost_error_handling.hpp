#ifndef __STAN__MATHS__BOOST_ERROR_HANDLING_HPP__
#define __STAN__MATHS__BOOST_ERROR_HANDLING_HPP__

namespace boost {

  namespace math {

    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(short n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(short n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(short n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(short n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned short n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(unsigned short n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(unsigned short n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned short n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(int n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(int n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(int n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(int n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned int n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(unsigned int n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(unsigned int n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned int n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(long n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(long n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(long n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(long n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned long n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(unsigned long n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(unsigned long n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(unsigned long n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(long long n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(long long n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(long long n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnormal(long long n) {
      return true;
    }


    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always finite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isfinite(unsigned long long n) {
      return true;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never infinite.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isinf(unsigned long long n) {
      return false;
    }
    /** 
     * Return <code>false</code> for all arguments.
     * Integer values are never NaN.
     * @param n Value to test.
     * @return <code>true</code>
     */
    template <>
    inline
    bool isnan(unsigned long long n) {
      return false;
    }
    /** 
     * Return <code>true</code> for all arguments.
     * Integer values are always normal.
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
