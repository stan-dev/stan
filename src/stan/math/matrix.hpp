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



    // scalar returns

        
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

