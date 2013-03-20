#ifndef __STAN__MATH__MATRIX_HPP__
#define __STAN__MATH__MATRIX_HPP__

#include <stdarg.h>
#include <stdexcept>
#include <ostream>
#include <vector>

#include <boost/math/tools/promotion.hpp>

#define EIGEN_DENSEBASE_PLUGIN "stan/math/EigenDenseBaseAddons.hpp"
#include <Eigen/Dense>
#include <Eigen/QR>

#include <stan/math/boost_error_handling.hpp>

namespace stan {
  
  namespace math {
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
    // from input type F to output type T 

    // scalar, F != T  (base template)
    template <typename F, typename T>
    struct promoter {
      inline static void promote(const F& u, T& t) {
        t = u;
      }
      inline static T promote_to(const F& u) {
        return u;
      }
    };
    // scalar, F == T
    template <typename T>
    struct promoter<T,T> {
      inline static void promote(const T& u, T& t) {
        t = u;
      }
      inline static T promote_to(const T& u) {
        return u;
      }
    };

    // std::vector, F != T
    template <typename F, typename T>
    struct promoter<std::vector<F>, std::vector<T> > {
      inline static void promote(const std::vector<F>& u,
                          std::vector<T>& t) {
        t.resize(u.size());
        for (size_t i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u[i],t[i]);
      }
      inline static std::vector<T>
      promote_to(const std::vector<F>& u) {
        std::vector<T> t;
        promoter<std::vector<F>,std::vector<T> >::promote(u,t);
        return t;
      }
    };
    // std::vector, F == T
    template <typename T>
    struct promoter<std::vector<T>, std::vector<T> > {
      inline static void promote(const std::vector<T>& u,
                          std::vector<T>& t) {
        t = u;
      }
      inline static std::vector<T> promote_to(const std::vector<T>& u) {
        return u;
      }
    };

    // Eigen::Matrix, F != T
    template <typename F, typename T, int R, int C>
    struct promoter<Eigen::Matrix<F,R,C>, Eigen::Matrix<T,R,C> > {
      inline static void promote(const Eigen::Matrix<F,R,C>& u,
                          Eigen::Matrix<T,R,C>& t) {
        t.resize(u.rows(), u.cols());
        for (int i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u(i),t(i));
      }
      inline static Eigen::Matrix<T,R,C>
      promote_to(const Eigen::Matrix<F,R,C>& u) {
        Eigen::Matrix<T,R,C> t;
        promoter<Eigen::Matrix<F,R,C>,Eigen::Matrix<T,R,C> >::promote(u,t);
        return t;
      }
    };
    // Eigen::Matrix, F == T
    template <typename T, int R, int C>
    struct promoter<Eigen::Matrix<T,R,C>, Eigen::Matrix<T,R,C> > {
      inline static void promote(const Eigen::Matrix<T,R,C>& u,
                          Eigen::Matrix<T,R,C>& t) {
        t = u;
      }
      inline static Eigen::Matrix<T,R,C> promote_to(const Eigen::Matrix<T,R,C>& u) {
        return u;
      }
    };

    template <typename T1, typename T2>
    struct common_type {
      typedef typename boost::math::tools::promote_args<T1,T2>::type type;
    };

    template <typename T1, typename T2>
    struct common_type<std::vector<T1>, std::vector<T2> > {
      typedef std::vector<typename common_type<T1,T2>::type> type;
    };
    
    template <typename T1, typename T2, int R, int C>
    struct common_type<Eigen::Matrix<T1,R,C>, Eigen::Matrix<T2,R,C> > {
      typedef Eigen::Matrix<typename common_type<T1,T2>::type,R,C> type;
    };

    template <typename T1, typename T2, typename F>
    inline
    typename common_type<T1,T2>::type
    promote_common(const F& u) {
      return promoter<F, typename common_type<T1,T2>::type>
        ::promote_to(u);
    }




    /**
     * Structure for building up arrays in an expression (rather than
     * in statements) using an argumentchaining add() method and 
     * a getter method array() to return the result.
     */
    template <typename T>
    struct array_builder {
      std::vector<T> x_;
      array_builder() : x_() { }
      template <typename F>
      array_builder& add(const F& u) {
        T t;
        promoter<F,T>::promote(u,t);
        x_.push_back(t);
        return *this;
      }
      std::vector<T> array() {
        return x_;
      }
    };

    /**
     * Type for matrix of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
    matrix_d;

    /**
     * Type for (column) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,1>
    vector_d;

    /**
     * Type for (row) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,1,Eigen::Dynamic>
    row_vector_d;

    namespace {

      template <typename T>
      void resize(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos],dims[pos+1]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
      }

      template <typename T>
      void resize(Eigen::Matrix<T,1,Eigen::Dynamic>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
      }


      void resize(double /*x*/, 
                  const std::vector<size_t>& /*dims*/, 
                  size_t /*pos*/) {
        // no-op
      }

      template <typename T>
      void resize(std::vector<T>& x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
        x.resize(dims[pos]);
        ++pos;
        if (pos >= dims.size()) return; // skips lowest loop to scalar
        for (size_t i = 0; i < x.size(); ++i)
          resize(x[i],dims,pos);
      }

    }

    /**
     * Recursively resize the specified vector of vectors,
     * which must bottom out at scalar values, Eigen vectors
     * or Eigen matrices.
     *
     * @param x Array-like object to resize.
     * @param dims New dimensions.
     * @tparam T Type of object being resized.
     */
    template <typename T>
    inline void resize(T& x, std::vector<size_t> dims) {
      resize(x,dims,0U);
    }

    // polymorphic gets with bounds checking


    namespace {
      
      void raise_range_error(size_t max,
                             size_t i, 
                             const char* msg,
                             size_t idx) {
          std::stringstream s;
          s << "INDEX OPERATOR [] OUT OF BOUNDS"
            << "; index=" << i
            << "; lower bound=1"
            << "; upper bound=" << max
            << "; index position=" << idx
            << "; " << msg
            << std::endl;
          throw std::out_of_range(s.str());
      }

      inline
      void check_range(size_t max,
                       size_t i, 
                       const char* msg,
                       size_t idx) {
#ifndef NDEBUG
        if (i < 1 || i > max) 
          raise_range_error(max,i,msg,idx);
#endif
      }

    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one index.  If the index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i Index into vector plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at <code>i - 1</code>
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<T>& x, 
                 size_t i, 
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i,error_msg,idx);
      return x[i - 1];
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<T> >& x, 
                 size_t i1, 
                 size_t i2,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<T> > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<std::vector<T> > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,i4,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,i4,i5,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,i4,i5,i6,error_msg,idx+1);
    }


    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param i7 Seventh index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 size_t i7,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,i4,i5,i6,i7,error_msg,idx+1);
    }


    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param i7 Seventh index plus 1.
     * @param i8 Eigth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 size_t i7,
                 size_t i8,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1(x[i1 - 1],i2,i3,i4,i5,i6,i7,i8,error_msg,idx+1);
    }



    /**
     * Return a copy of the row of the specified vector at the specified
     * base-one row index.  If the index is out of range, throw a
     * <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * <b>Warning</b>:  Because a copy is involved, it is inefficient
     * to access element of matrices by first using this method
     * to get a row then using a second call to get the value at 
     a specified column.
     *
     * @param x Matrix from which to get a row
     * @param m Index into matrix plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Row of matrix at <code>i - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    get_base1(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,
              size_t m,
              const char* error_msg,
              size_t idx) {
      check_range(x.rows(),m,error_msg,idx);
      return x.block(m-1,0,1,x.cols());
    }

    /**
     * Return a reference to the value of the specified matrix at the specified
     * base-one row and column indexes.  If either index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Matrix from which to get a row
     * @param m Row index plus 1.
     * @param n Column index plus 1.
     * @param error_msg Error message if either index is out of range.
     * @param idx Nested index level to report in error message if
     * either index is out of range.
     * @return Value of matrix at row <code>m - 1</code> and column
     * <code>n - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,
                 size_t m,
                 size_t n,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.rows(),m,error_msg,idx);
      check_range(x.cols(),n,error_msg,idx + 1);
      return x(m - 1, n - 1);
    }

    /**
     * Return a reference to the value of the specified column vector
     * at the specified base-one index.  If the index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Column vector from which to get a value.
     * @param m Row index plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of column vector at row <code>m - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                 size_t m,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),m,error_msg,idx);
      return x(m - 1);
      
    }

    /**
     * Return a reference to the value of the specified row vector
     * at the specified base-one index.  If the index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Row vector from which to get a value.
     * @param n Column index plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of row vector at column <code>n - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1(Eigen::Matrix<T,1,Eigen::Dynamic>& x,
                 size_t n,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),n,error_msg,idx);
      return x(n - 1);
    }


    // ********************** start lhs

   /**
     * Return a reference to the value of the specified vector at the
     * specified base-one index.  If the index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i Index into vector plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at <code>i - 1</code>
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<T>& x, 
                 size_t i, 
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i,error_msg,idx);
      return x[i - 1];
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<T> >& x, 
                 size_t i1, 
                 size_t i2,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<T> > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<std::vector<T> > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,i4,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,i4,i5,error_msg,idx+1);
    }

    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,i4,i5,i6,error_msg,idx+1);
    }


    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param i7 Seventh index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 size_t i7,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,i4,i5,i6,i7,error_msg,idx+1);
    }


    /**
     * Return a reference to the value of the specified vector at the
     * specified base-one indexes.  If an index is out of range, throw
     * a <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * @param x Vector from which to get a value.
     * @param i1 First index plus 1.
     * @param i2 Second index plus 1.
     * @param i3 Third index plus 1.
     * @param i4 Fourth index plus 1.
     * @param i5 Fifth index plus 1.
     * @param i6 Sixth index plus 1.
     * @param i7 Seventh index plus 1.
     * @param i8 Eigth index plus 1.
     * @param error_msg Error message if an index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of vector at indexes.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > > > >& x, 
                 size_t i1, 
                 size_t i2,
                 size_t i3,
                 size_t i4,
                 size_t i5,
                 size_t i6,
                 size_t i7,
                 size_t i8,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),i1,error_msg,idx);
      return get_base1_lhs(x[i1 - 1],i2,i3,i4,i5,i6,i7,i8,error_msg,idx+1);
    }



    /**
     * Return a copy of the row of the specified vector at the specified
     * base-one row index.  If the index is out of range, throw a
     * <code>std::out_of_range</code> exception with the specified
     * error message and index indicated.
     *
     * <b>Warning</b>:  Because a copy is involved, it is inefficient
     * to access element of matrices by first using this method
     * to get a row then using a second call to get the value at 
     a specified column.
     *
     * @param x Matrix from which to get a row
     * @param m Index into matrix plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Row of matrix at <code>i - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    Eigen::Block<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
    get_base1_lhs(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,
              size_t m,
              const char* error_msg,
              size_t idx) {
      check_range(x.rows(),m,error_msg,idx);
      return x.block(m-1,0,1,x.cols());
    }

    /**
     * Return a reference to the value of the specified matrix at the specified
     * base-one row and column indexes.  If either index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Matrix from which to get a row
     * @param m Row index plus 1.
     * @param n Column index plus 1.
     * @param error_msg Error message if either index is out of range.
     * @param idx Nested index level to report in error message if
     * either index is out of range.
     * @return Value of matrix at row <code>m - 1</code> and column
     * <code>n - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,
                 size_t m,
                 size_t n,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.rows(),m,error_msg,idx);
      check_range(x.cols(),n,error_msg,idx + 1);
      return x(m - 1, n - 1);
    }

    /**
     * Return a reference to the value of the specified column vector
     * at the specified base-one index.  If the index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Column vector from which to get a value.
     * @param m Row index plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of column vector at row <code>m - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                 size_t m,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),m,error_msg,idx);
      return x(m - 1);
      
    }

    /**
     * Return a reference to the value of the specified row vector
     * at the specified base-one index.  If the index is out of range,
     * throw a <code>std::out_of_range</code> exception with the
     * specified error message and index indicated.
     *
     * @param x Row vector from which to get a value.
     * @param n Column index plus 1.
     * @param error_msg Error message if the index is out of range.
     * @param idx Nested index level to report in error message if
     * the index is out of range.
     * @return Value of row vector at column <code>n - 1</code>.
     * @tparam T type of value.
     */
    template <typename T>
    inline
    T& get_base1_lhs(Eigen::Matrix<T,1,Eigen::Dynamic>& x,
                 size_t n,
                 const char* error_msg,
                 size_t idx) {
      check_range(x.size(),n,error_msg,idx);
      return x(n - 1);
    }



    // int returns

    template <typename T, int R, int C>
    inline 
    size_t 
    rows(const Eigen::Matrix<T,R,C>& m) {
      return m.rows();
    }
    template <typename T, int R, int C>
    inline 
    size_t 
    cols(const Eigen::Matrix<T,R,C>& m) {
      return m.cols();
    }

    template <typename T1, typename T2>
    inline
    void validate_less_or_equal(const T1& x, const T2& y,
                                const char* x_name, const char* y_name, 
                                const char* fun_name) {
      if (x <= y) return;
      std::stringstream ss;
      ss << "require " << x_name << " <= " << y_name
         << " in " << fun_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }
    template <typename T1, typename T2>
    inline
    void validate_less(const T1& x, const T2& y,
                       const char* x_name, const char* y_name, 
                       const char* fun_name) {
      if (x < y) return;
      std::stringstream ss;
      ss << "require " << x_name << " < " << y_name
         << " in " << fun_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }
    template <typename T1, typename T2>
    inline
    void validate_greater_or_equal(const T1& x, const T2& y,
                                   const char* x_name, const char* y_name, 
                                   const char* fun_name) {
      if (x >= y) return;
      std::stringstream ss;
      ss << "require " << x_name << " >= " << y_name
         << " in " << fun_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }
    template <typename T1, typename T2>
    inline
    void validate_greater(const T1& x, const T2& y,
                          const char* x_name, const char* y_name, 
                          const char* fun_name) {
      if (x > y) return;
      std::stringstream ss;
      ss << "require " << x_name << " > " << y_name
         << " in " << fun_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }



    template <typename T, int R, int C>
    void validate_column_index(const Eigen::Matrix<T,R,C>& m,
                               size_t j,
                               const char* msg) {
      if (j > 0 && j <=  static_cast<size_t>(m.cols())) return;
      std::stringstream ss;
      ss << "require 0 < column index <= number of columns in" << msg;
      ss << " found cols()=" << m.cols()
         << "; index j=" << j;
      throw std::domain_error(ss.str());
    }

    template <typename T, int R, int C>
    void validate_row_index(const Eigen::Matrix<T,R,C>& m,
                            size_t i,
                            const char* msg) {
      if (i > 0 && i <=  static_cast<size_t>(m.rows())) return;
      std::stringstream ss;
      ss << "require 0 < row index <= number of rows in" << msg;
      ss << " found rows()=" << m.rows()
         << "; index i=" << i;
      throw std::domain_error(ss.str());
    }
    
    template <typename T, int R, int C>
    void validate_square(const Eigen::Matrix<T,R,C>& x,
                         const char* msg) {
      if (x.rows() == x.cols()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require square matrix, but found"
         << " rows=" << x.rows()
         << "; cols=" << x.cols();
      throw std::domain_error(ss.str());
    }

    template <typename T, int R, int C>
    void validate_symmetric(const Eigen::Matrix<T,R,C>& x,
                            const char* msg) {
      // tolerance = 1E-8
      validate_square(x,msg);
      for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
          if (x(i,j) != x(j,i)) {
            std::stringstream ss;
            ss << "error in call to " << msg
               << "; require symmetric matrix, but found"
               << "; x[" << i << "," << j << "]=" << x(i,j)
               << "; x[" << j << "," << i << "]=" << x(j,i);
            throw std::domain_error(ss.str());
          }
        }
      }    
    }

    template <typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline void validate_matching_dims(const Eigen::Matrix<T1,R1,C1>& x1,
                                       const Eigen::Matrix<T2,R2,C2>& x2,
                                       const char* msg) {
      if (x1.rows() == x2.rows()
          && x1.cols() == x2.cols()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching dimensions, but found"
         << " arg1(rows=" << x1.rows() << ",cols=" << x1.cols() << ");"
         << " arg2(rows=" << x2.rows() << ",cols=" << x2.cols() << ")";
      throw std::domain_error(ss.str());
    }

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



    template <typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline void validate_multiplicable(const Eigen::Matrix<T1,R1,C1>& x1,
                                       const Eigen::Matrix<T2,R2,C2>& x2,
                                       const char* msg) {
      if (x1.cols() == x2.rows()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require cols of arg1 to match rows of arg2, but found "
         << " arg1 rows=" << x1.rows() << " arg1 cols=" << x1.cols()
         << " arg2 rows=" << x2.rows() << " arg2 cols=" << x2.cols();
      throw std::domain_error(ss.str());
    }    

    template <typename T>
    inline void validate_nonzero_size(const T& x, const char* msg) {
      if (x.size() > 0) return;
      std::stringstream ss;
      ss << "require non-zero size for " << msg
         << "found size=" << x.size();
      throw std::domain_error(ss.str());
    }

    template <typename T, int R, int C>
    inline void validate_vector(const Eigen::Matrix<T,R,C>& x,
                                const char* msg) {
      if (x.rows() == 1 || x.cols() == 1) return;
      std::stringstream ss;
      ss << "error in " << msg
         << "; require vector, found "
         << " rows=" << x.rows() << "cols=" << x.cols();
      throw std::domain_error(ss.str());
    }    

    // scalar returns

    /**
     * Returns the determinant of the specified square matrix.
     *
     * @param m Specified matrix.
     * @return Determinant of the matrix.
     * @throw std::domain_error if matrix is not square.
     */
    template <typename T,int R, int C>
    inline T determinant(const Eigen::Matrix<T,R,C>& m) {
      stan::math::validate_square(m,"determinant");
      return m.determinant();
    }
    
    /**
     * Returns the log absolute determinant of the specified square matrix.
     *
     * @param m Specified matrix.
     * @return log absolute determinant of the matrix.
     * @throw std::domain_error if matrix is not square.
     */
    template <typename T,int R, int C>
    inline T log_determinant(const Eigen::Matrix<T,R,C>& m) {
      stan::math::validate_square(m,"log_determinant");
      return m.colPivHouseholderQr().logAbsDeterminant();
    }
    

    /**
     * Returns the dot product of the specified vector with itself.
     * @param v Vector.
     * @tparam R number of rows or <code>Eigen::Dynamic</code> for dynamic
     * @tparam C number of rows or <code>Eigen::Dyanmic</code> for dynamic
     * @throw std::domain_error If v is not vector dimensioned.
     */
    template <int R, int C>
    inline double dot_self(const Eigen::Matrix<double, R, C>& v) {
      validate_vector(v,"dot_self");
      double sum = 0.0;
      for (int i = 0; i < v.size(); ++i)
        sum += v(i) * v(i);
      return sum;
    }

    /**
     * Returns the dot product of each column of a matrix with itself.
     * @param x Matrix.
     * @tparam T scalar type
     */
    template<typename T,int R,int C>
    inline Eigen::Matrix<T,1,C> 
    columns_dot_self(const Eigen::Matrix<T,R,C>& x) {
      return x.colwise().squaredNorm();
    }

    /**
     * Returns the dot product of the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<int R1,int C1,int R2, int C2>
    inline Eigen::Matrix<double, 1, C1>
    columns_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                        const Eigen::Matrix<double, R2, C2>& v2) {
      validate_matching_sizes(v1,v2,"columns_dot_product");
      Eigen::Matrix<double, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = v1.col(j).dot(v2.col(j));
      }
      return ret;
    }

    /**
     * Returns the dot product of the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<int R1,int C1,int R2, int C2>
    inline double dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                              const Eigen::Matrix<double, R2, C2>& v2) {
      validate_vector(v1,"dot_product");
      validate_vector(v2,"dot_product");
      validate_matching_sizes(v1,v2,"dot_product");
      double sum = 0.0;
      for (int i = 0; i < v1.size(); ++i)
        sum += v1[i] * v2[i]; 
      return sum;
    }
    /**
     * Returns the dot product of the specified arrays of doubles.
     * @param v1 First array.
     * @param v2 Second array.
     * @param length Length of both arrays.
     */
    inline double dot_product(const double* v1, const double* v2, 
                              size_t length) {
      double result = 0;
      for (size_t i = 0; i < length; i++)
        result += v1[i] * v2[i];
      return result;
    }
    /**
     * Returns the dot product of the specified arrays of doubles.
     * @param v1 First array.
     * @param v2 Second array.
     * @throw std::domain_error if the vectors are not the same size.
     */
    inline double dot_product(const std::vector<double>& v1,
                              const std::vector<double>& v2) {
      validate_matching_sizes(v1,v2,"dot_product");
      return dot_product(&v1[0], &v2[0], v1.size());
    }

    /**
     * Returns the minimum coefficient in the specified
     * column vector.
     * @param v Specified vector.
     * @return Minimum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     */
    inline int min(const std::vector<int>& x) {
      if (x.size() == 0)
        throw std::domain_error("error: cannot take min of empty int vector");
      int min = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] < min) 
          min = x[i];
      return min;
    }

    /**
     * Returns the minimum coefficient in the specified
     * column vector.
     * @param v Specified vector.
     * @return Minimum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     */
    template <typename T>
    inline T min(const std::vector<T>& x) {
      if (x.size() == 0)
        return std::numeric_limits<T>::infinity();
      T min = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] < min) 
          min = x[i];
      return min;
    }

    /**
     * Returns the minimum coefficient in the specified
     * matrix, vector, or row vector.
     * @param v Specified matrix, vector, or row vector.
     * @return Minimum coefficient value in the vector.
     */
    template <typename T, int R, int C>
    inline T min(const Eigen::Matrix<T,R,C>& m) {
      if (m.size() == 0) 
        return std::numeric_limits<double>::infinity();
      return m.minCoeff();
    }

    // vector and matrix returns

    // void eigen_decompose_sym(const matrix_d& m,
    //                          vector_d& eigenvalues,
    //                          matrix_d& eigenvectors) {
    //   Eigen::SelfAdjointEigenSolver<matrix_d> solver(m);
    //   eigenvalues = solver.eigenvalues().real();
    //   eigenvectors = solver.eigenvectors().real();
    // }


    // void svd(const matrix_d& m, matrix_d& u, matrix_d& v, vector_d& s) {
    //   static const unsigned int THIN_SVD_OPTIONS
    //     = Eigen::ComputeThinU | Eigen::ComputeThinV;
    //   Eigen::JacobiSVD<matrix_d> svd(m, THIN_SVD_OPTIONS);
    //   u = svd.matrixU();
    //   v = svd.matrixV();
    //   s = svd.singularValues();
    // }



  }
}
#endif

