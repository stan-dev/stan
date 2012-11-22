#ifndef __STAN__MATH__MATRIX_HPP__
#define __STAN__MATH__MATRIX_HPP__

#include <stdarg.h>
#include <stdexcept>
#include <ostream>
#include <vector>

#include <boost/math/tools/promotion.hpp>

#define EIGEN_DENSEBASE_PLUGIN "stan/math/EigenDenseBaseAddons.hpp"
#include <Eigen/Dense>

#include <stan/math/boost_error_handling.hpp>

namespace stan {
  
  namespace math {


    // from input type F to output type T 

    // scalar, F != T  (base template)
    template <typename F, typename T>
    struct promoter {
      static void promote(const F& u, T& t) {
        t = u;
      }
    };
    // scalar, F == T
    template <typename T>
    struct promoter<T,T> {
      static void promote(const T& u, T& t) {
        t = u;
      }
    };

    // std::vector, F != T
    template <typename F, typename T>
    struct promoter<std::vector<F>, std::vector<T> > {
      static void promote(const std::vector<F>& u,
                          std::vector<T>& t) {
        t.resize(u.size());
        for (size_t i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u[i],t[i]);
      }
    };
    // F = T, std::vector
    template <typename T>
    struct promoter<std::vector<T>, std::vector<T> > {
      static void promote(const std::vector<T>& u, std::vector<T>& t) {
        t = u;
      }
    };

    // Eigen::Matrix, F != T
    template <typename F, typename T, int R, int C>
    struct promoter<Eigen::Matrix<F,R,C>, Eigen::Matrix<T,R,C> > {
      static const std::vector<T> promote(const Eigen::Matrix<F,R,C>& u,
                                          Eigen::Matrix<T,R,C>& t) {
        t.resize(u.rows(), u.cols());
        for (size_t i = 0; i < u.size(); ++i)
          promoter<F,T>::promote(u(i),t(i));
      }
    };
    // Eigen::Matrix, F == T
    template <typename T, int R, int C>
    struct promoter<Eigen::Matrix<T,R,C>, Eigen::Matrix<T,R,C> > {
      static const std::vector<T> promote(const Eigen::Matrix<T,R,C>& u,
                                          Eigen::Matrix<T,R,C>& t) {
        t = u;
      }
    };


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

    // FIXME:  remove this as it doesn't support type inference
    // through the definitions

    /**
     * This is a traits structure for Eigen matrix types.
     *
     * @tparam T Underlying scalar type.
     */ 
    template<typename T>
    struct EigenType {

      /** Type of scalar. 
       */
      typedef T scalar;

      /**
       * Type of Eigen matrix.
       */
      typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>  matrix;

      /**
       * Type of Eigen column vector.
       */
      typedef Eigen::Matrix<T,Eigen::Dynamic,1>  vector;

      /**
       * Type of Eigen row vector.
       */
      typedef Eigen::Matrix<T,1,Eigen::Dynamic>  row_vector;

      /**
       * Type of Eigen diagonal matrix.
       */
      typedef Eigen::DiagonalMatrix<T,Eigen::Dynamic> diagonal_matrix;

      /**
       * Type of Eigen array shaped like a (column) vector.
       */
      typedef Eigen::Array<T,1,Eigen::Dynamic> array_vector;
      
      /**
       * Type of Eigen array shaped like a row vector.
       */
      typedef Eigen::Array<T,Eigen::Dynamic,1> array_row_vector;

      /**
       * Type of Eigen array shaped like a matrix.
       */
      typedef Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic> array_matrix;
    };      
    
    /**
     * Type for matrix of double values.
     */
    typedef EigenType<double>::matrix matrix_d;

    /**
     * Type for (column) vector of double values.
     */
    typedef EigenType<double>::vector vector_d;

    /**
     * Type for (row) vector of double values.
     */
    typedef EigenType<double>::row_vector row_vector_d;

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


      void resize(double x, 
                  const std::vector<size_t>& dims, 
                  size_t pos) {
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
      return x.row(m - 1);
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

    template <typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline void validate_matching_dims(const Eigen::Matrix<T1,R1,C1>& x1,
                                       const Eigen::Matrix<T2,R2,C2>& x2,
                                       const char* msg) {
      if (x1.rows() == x2.rows()
          && x1.cols() == x2.cols()) return;
      std::stringstream ss;
      ss << "error in call to " << msg
         << "; require matching dimensions, but found"
         << " arg1 rows=" << x1.rows() << " arg1 cols=" << x1.cols()
         << " arg2 rows=" << x2.rows() << " arg2 cols=" << x2.cols();
      throw std::domain_error(ss.str());
    }

    template <typename T1, typename T2>
    inline void validate_matching_sizes(const std::vector<T1>& x1,
                                        const std::vector<T2>& x2,
                                        const char* msg) {
      if (x1.size() == x2.size()) return;
      std::stringstream ss;
      ss << "require matching sizes in " << msg
         << " found first argument size=" << x1.size()
         << "; second argument size=" << x2.size();
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
         << " arg1 rows=" << x1.rows() << " arg1 cols=" << x1.cols()
         << " arg1 size=" << (x1.rows() * x1.cols())
         << " arg2 rows=" << x2.rows() << " arg2 cols=" << x2.cols()
         << " arg2 size=" << (x2.rows() * x2.cols());
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
     * @param m Specified matrix.
     * @return Determinant of the matrix.
     * @throw std::domain_error if matrix is not square.
     */
    double determinant(const matrix_d& m);


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
    template<typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,1> 
    columns_dot_self(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x) {
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



    /**
     * Returns the maximum coefficient in the specified
     * column vector.
     * @param v Specified vector.
     * @return Maximum coefficient value in the vector.
     * @tparam Type of values being compared and returned
     * @throw std::domain_error If the size of the vector is zero.
     */
    inline int max(const std::vector<int>& x) {
      if (x.size() == 0)
        throw std::domain_error("error: cannot take max of empty int vector");
      int max = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > max) 
          max = x[i];
      return max;
    }

    /**
     * Returns the maximum coefficient in the specified
     * column vector.
     * @param v Specified vector.
     * @return Maximum coefficient value in the vector.
     * @tparam T Type of values being compared and returned
     */
    template <typename T>
    inline T max(const std::vector<T>& x) {
      if (x.size() == 0)
        return -std::numeric_limits<T>::infinity();
      T max = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > max) 
          max = x[i];
      return max;
    }

    /**
     * Returns the maximum coefficient in the specified
     * vector, row vector, or matrix.
     * @param v Specified vector, row vector, or matrix.
     * @return Maximum coefficient value in the vector.
     */
    template <typename T, int R, int C>
    inline T max(const Eigen::Matrix<T,R,C>& m) {
      if (m.size() == 0)
        return -std::numeric_limits<double>::infinity();
      return m.maxCoeff();
    }

    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified standard vector.
     * @param v Specified vector.
     * @return Sample mean of vector coefficients.
     * @throws std::domain_error if the size of the vector is less
     * than 1.
     */
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    mean(const std::vector<T>& v) {
      validate_nonzero_size(v,"mean");
      T sum(v[0]);
      for (size_t i = 1; i < v.size(); ++i)
        sum += v[i];
      return sum / v.size();
    }

    /**
     * Returns the sample mean (i.e., average) of the coefficients
     * in the specified vector, row vector, or matrix.
     * @param m Specified vector, row vector, or matrix.
     * @return Sample mean of vector coefficients.
     */
    template <typename T, int R, int C>
    inline  
    typename boost::math::tools::promote_args<T>::type
    mean(const Eigen::Matrix<T,R,C>& m) {
      validate_nonzero_size(m,"mean");
      return m.mean();
    }

    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified standard vector.
     * @param v Specified vector.
     * @return Sample variance of vector.
     * @throws std::domain_error if the size of the vector is less
     * than 1.
     */
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    variance(const std::vector<T>& v) {
      validate_nonzero_size(v,"variance");
      if (v.size() == 1)
        return 0.0;
      T v_mean(mean(v));
      T sum_sq_diff(0);
      for (size_t i = 0; i < v.size(); ++i) {
        T diff = v[i] - v_mean;
        sum_sq_diff += diff * diff;
      }
      return sum_sq_diff / (v.size() - 1);
    }

    /**
     * Returns the sample variance (divide by length - 1) of the
     * coefficients in the specified column vector.
     * @param v Specified vector.
     * @return Sample variance of vector.
     */
    template <typename T, int R, int C>
    inline
    typename boost::math::tools::promote_args<T>::type
    variance(const Eigen::Matrix<T,R,C>& m) {
      validate_nonzero_size(m,"variance");
      if (m.size() == 1)
        return 0.0;
      typename boost::math::tools::promote_args<T>::type 
        mn(mean(m));
      typename boost::math::tools::promote_args<T>::type 
        sum_sq_diff(0);
      for (int i = 0; i < m.size(); ++i) {
        typename boost::math::tools::promote_args<T>::type 
          diff = m(i) - mn;
        sum_sq_diff += diff * diff;
      }
      return sum_sq_diff / (m.size() - 1);
    }

    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified column vector.
     * @param v Specified vector.
     * @return Sample variance of vector.
     */
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    sd(const std::vector<T>& v) {
      validate_nonzero_size(v,"sd");
      if (v.size() == 1) return 0.0;
      return sqrt(variance(v));
    }

    /**
     * Returns the unbiased sample standard deviation of the
     * coefficients in the specified vector, row vector, or matrix.
     * @param m Specified vector, row vector or matrix.
     * @return Sample variance.
     */
    template <typename T, int R, int C>
    inline 
    typename boost::math::tools::promote_args<T>::type
    sd(const Eigen::Matrix<T,R,C>& m) {
      // FIXME: redundant with test in variance; second line saves sqrt
      validate_nonzero_size(m,"sd");  
      if (m.size() == 1) return 0.0;
      return sqrt(variance(m));
    }

    /**
     * Return the sum of the values in the specified
     * standard vector.
     *
     * @param xs Standard vector to sum.
     * @return Sum of elements.
     * @tparam T Type of elements summed.
     */
    template <typename T>
    inline T sum(const std::vector<T>& xs) {
      if (xs.size() == 0) return 0;
      T sum(xs[0]);
      for (size_t i = 1; i < xs.size(); ++i)
        sum += xs[i];
      return sum;
    }

    /**
     * Returns the sum of the coefficients of the specified
     * column vector.
     * @param v Specified vector.
     * @return Sum of coefficients of vector.
     */
    template <typename T, int R, int C>
    inline double sum(const Eigen::Matrix<T,R,C>& v) {
      return v.sum();
    }

    /**
     * Returns the product of the coefficients of the specified
     * standard vector.
     * @param v Specified vector.
     * @return Product of coefficients of vector.
     */
    template <typename T>
    inline T prod(const std::vector<T>& v) {
      if (v.size() == 0) return 1;
      T product = v[0];
      for (size_t i = 1; i < v.size(); ++i)
        product *= v[i];
      return product;
    }
    
    /**
     * Returns the product of the coefficients of the specified
     * column vector.
     * @param v Specified vector.
     * @return Product of coefficients of vector.
     */
    template <typename T, int R, int C>
    inline T prod(const Eigen::Matrix<T,R,C>& v) {
      if (v.size() == 0) return 1.0;
      return v.prod();
    }

    /**
     * Returns the trace of the specified matrix.  The trace
     * is defined as the sum of the elements on the diagonal.
     * The matrix is not required to be square.  Returns 0 if
     * matrix is empty.
     *
     * @param[in] m Specified matrix.
     * @return Trace of the matrix.
     */
    template <typename T>
    inline T trace(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      return m.trace();
    }

    /**
     * Return the element-wise logarithm of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i,j) = log(m(i,j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T,Rows,Cols> log(const Eigen::Matrix<T,Rows,Cols>& m) {
      return m.array().log().matrix();
    }

    /**
     * Return the element-wise exponentiation of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i,j) = exp(m(i,j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T,Rows,Cols> exp(const Eigen::Matrix<T,Rows,Cols>& m) {
      return m.array().exp().matrix();
    }


    // vector and matrix returns

    /**
      * Return the sum of the specified matrices.  The two matrices
      * must have the same dimensions. 

      * @tparam T1 Scalar type of first matrix.
      * @tparam T2 Scalar type of second matrix.
      * @tparam R Row type of matrices.
      * @tparam C Column type of matrices.
      * @param m1 First matrix.
      * @param m2 Second matrix.  
      * @return Sum of the matrices.
      * @throw std::domain_error if m1 and m2 do not have the same
      * dimensions.
      */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const Eigen::Matrix<T1,R,C>& m1,
        const Eigen::Matrix<T2,R,C>& m2) {
      stan::math::validate_matching_dims(m1,m2,"add");
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m1.rows(),m1.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m1(i) + m2(i);
      return result;
    }
    /**
     * Return the sum of the specified matrix and specified scalar.
     *
     * @tparam T1 Scalar type of matrix.
     * @param T2 Type of scalar.
     * @param m Matrix.
     * @param c Scalar.
     * @return The matrix plus the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const Eigen::Matrix<T1,R,C>& m, 
        const T2& c) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m.rows(),m.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m(i) + c;
      return result;
    }
    /**
     * Return the sum of the specified scalar and specified matrix.
     *
     * @param T1 Type of scalar.
     * @tparam T2 Scalar type of matrix.
     * @param c Scalar.
     * @param m Matrix.
     * @return The scalar plus the matrix.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const T1& c,
        const Eigen::Matrix<T2,R,C>& m) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m.rows(),m.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = c + m(i);
      return result;
    }


    /**
     * Return the result of subtracting the second specified matrix
     * from the first specified matrix.  The return scalar type is the
     * promotion of the input types.
     *
     * @tparam T1 Scalar type of first matrix.
     * @tparam T2 Scalar type of second matrix.
     * @tparam R Row type of matrices.
     * @tparam C Column type of matrices.
     * @param m1 First matrix.
     * @param m2 Second matrix.
     * @return Difference between first matrix and second matrix.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1,R,C>& m1,
             const Eigen::Matrix<T2,R,C>& m2) {
      stan::math::validate_matching_dims(m1,m2,"subtract");
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m1.rows(), m1.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m1(i) - m2(i);
      return result;
    }

    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const T1& c,
             const Eigen::Matrix<T2,R,C>& m) {
      
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        result(i) = c - m(i);
      return result;
    }
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1,R,C>& m,
             const T2& c) {
      
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        result(i) = m(i) - c;
      return result;
    }



    /**
     * Returns the negation of the specified scalar or matrix.
     *
     * @tparam T Type of subtrahend.
     * @param x Subtrahend.
     * @return Negation of subtrahend.
     */
    template <typename T>
    inline
    T minus(const T& x) {
      return -x;
    }


    /**
     * Return specified matrix divided by specified scalar.
     * @tparam R Row type for matrix.
     * @tparam C Column type for matrix.
     * @param m Matrix.
     * @param c Scalar.
     * @return Matrix divided by scalar.
     */
    template <int R, int C>
    inline
    Eigen::Matrix<double,R,C>
    divide(const Eigen::Matrix<double,R,C>& m,
           double c) {
      return m / c;
    }

    /**
     * Return the elementwise multiplication of the specified
     * matrices.  
     *
     * @tparam T1 Type of scalars in first matrix.
     * @tparam T2 Type of scalars in second matrix.
     * @tparam R Row type of both matrices.
     * @tparam C Column type of both matrices.
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Elementwise product of matrices.
     */
    template <typename T1, typename T2, int R, int C>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    elt_multiply(const Eigen::Matrix<T1,R,C>& m1,
                 const Eigen::Matrix<T2,R,C>& m2) {
      stan::math::validate_matching_dims(m1,m2,"elt_multiply");
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.size(); ++i)
        result(i) = m1(i) * m2(i);
      return result;
    }


    /**
     * Return the elementwise division of the specified matrices
     * matrices.  
     *
     * @tparam T1 Type of scalars in first matrix.
     * @tparam T2 Type of scalars in second matrix.
     * @tparam R Row type of both matrices.
     * @tparam C Column type of both matrices.
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Elementwise division of matrices.
     */
    template <typename T1, typename T2, int R, int C>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    elt_divide(const Eigen::Matrix<T1,R,C>& m1,
               const Eigen::Matrix<T2,R,C>& m2) {
      stan::math::validate_matching_dims(m1,m2,"elt_multiply");
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.size(); ++i)
        result(i) = m1(i) / m2(i);
      return result;
    }
        

    /**
     * Return specified matrix multiplied by specified scalar.
     * @tparam R Row type for matrix.
     * @tparam C Column type for matrix.
     * @param m Matrix.
     * @param c Scalar.
     * @return Product of matrix and scalar.
     */
    template <int R, int C>
    inline
    Eigen::Matrix<double,R,C>
    multiply(const Eigen::Matrix<double,R,C>& m,
             double c) {
      return c * m;
    }
    /**
     * Return specified scalar multiplied by specified matrix.
     * @tparam R Row type for matrix.
     * @tparam C Column type for matrix.
     * @param c Scalar.
     * @param m Matrix.
     * @return Product of scalar and matrix.
     */
    template <int R, int C>
    inline
    Eigen::Matrix<double,R,C>
    multiply(double c,
             const Eigen::Matrix<double,R,C>& m) {
      return c * m;
    }


    /**
     * Return the product of the specified matrices.  The number of
     * columns in the first matrix must be the same as the number of rows
     * in the second matrix.
     * @param m1 First matrix.
     * @param m2 Second matrix.
     * @return The product of the first and second matrices.
     * @throw std::domain_error if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> multiply(const Eigen::Matrix<double,R1,C1>& m1,
                                                const Eigen::Matrix<double,R2,C2>& m2) {
      
      if (m1.cols() != m2.rows())
        throw std::domain_error("for multiplications, rows and columns must match; found m1.cols() != m2.rows()");
      return m1*m2;
    }
    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param rv Row vector.
     * @param v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::domain_error if rv and v are not the same size.
     */
    template<int C1,int R2>
    inline double multiply(const Eigen::Matrix<double,1,C1>& rv,
                           const Eigen::Matrix<double,R2,1>& v) {
      stan::math::validate_matching_sizes(rv,v,"multiply");
      if (rv.size() != v.size()) 
        throw std::domain_error("rv.size() != v.size()");
      return rv.dot(v);
    }

    /**
     * Returns the result of multiplying the lower triangular
     * portion of the square input matrix by its own transpose.
     * @param L Matrix to multiply.
     * @return The lower triangular values in L times their own
     * transpose.
     * @throw std::domain_error If the input matrix is not square.
     */
    inline matrix_d
    multiply_lower_tri_self_transpose(const matrix_d& L) {
      stan::math::validate_square(L,"multiply_lower_tri_self_transpose");
      if (L.rows() == 0)
        return matrix_d(0,0);
      if (L.rows() == 1) {
        matrix_d result(1,1);
        result(0,0) = L(0,0) * L(0,0);
        return result;
      }
      // FIXME:  write custom following agrad/matrix because can't get L_tri into
      // multiplication as no template support for tri * tri
      matrix_d L_tri = L.transpose().triangularView<Eigen::Upper>();
      return L.triangularView<Eigen::Lower>() * L_tri;
    }

    /**
     * Returns the result of post-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return M times its transpose.
     */
    inline matrix_d
    tcrossprod(const matrix_d& M) {
        if(M.rows() == 0)
          return matrix_d(0,0);
        if(M.rows() == 1) {
          return M * M.transpose();
        }
        matrix_d result(M.rows(),M.rows());
        return result.setZero().selfadjointView<Eigen::Upper>().rankUpdate(M);
    }

    /**
     * Returns the result of pre-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return Transpose of M times M
     */
    inline matrix_d
    crossprod(const matrix_d& M) {
        return tcrossprod(M.transpose());
    }

    /**
     * Return the specified row of the specified matrix, using
     * start-at-1 indexing.  
     *
     * This is equivalent to calling <code>m.row(i - 1)</code> and
     * assigning the resulting template expression to a row vector.
     * 
     * @param m Matrix.
     * @param i Row index.
     * @return Specified row of the matrix.
     */
    row_vector_d row(const matrix_d& m, size_t i);

    /**
     * Return the specified column of the specified matrix
     * using start-at-1 indexing.
     *
     * This is equivalent to calling <code>m.col(i - 1)</code> and
     * assigning the resulting template expression to a column vector.
     * 
     * @param m Matrix.
     * @param j Column index.
     * @return Specified column of the matrix.
     */
    vector_d col(const matrix_d& m, size_t j);

    /**
     * Return a column vector of the diagonal elements of the
     * specified matrix.  The matrix is not required to be square.
     * @param m Specified matrix.  
     * @return Diagonal of the matrix.
     */
    vector_d diagonal(const matrix_d& m);

    /**
     * Return a square diagonal matrix with the specified vector of
     * coefficients as the diagonal values.
     * @param v Specified vector.
     * @return Diagonal matrix with vector as diagonal values.
     */
    matrix_d diag_matrix(const vector_d& v);

    /**
     * Return the transposition of the specified column
     * vector.
     * @param v Specified vector.
     * @return Transpose of the vector.
     */
    row_vector_d transpose(const vector_d& v);
    /**
     * Return the transposition of the specified row
     * vector.
     * @param rv Specified vector.
     * @return Transpose of the vector.
     */
    vector_d transpose(const row_vector_d& rv);
    /**
     * Return the transposition of the specified matrix.
     * @param m Specified matrix.
     * @return Transpose of the matrix.
     */
    matrix_d transpose(const matrix_d& m);

    /**
     * Returns the inverse of the specified matrix.
     * @param m Specified matrix.
     * @return Inverse of the matrix.
     */
    matrix_d inverse(const matrix_d& m);

    /**
     * Return the softmax of the specified vector.
     * @param y Vector to transform
     * @return Unit simplex result of the softmax transform of the vector.
     */
    vector_d softmax(const vector_d& y);
    

    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_left_tri_low(const Eigen::Matrix<double,R1,C1> &A,
                                                            const Eigen::Matrix<double,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      if (A.cols() != b.rows())
        throw std::domain_error("A.cols() != b.rows()");
      return A.template triangularView<Eigen::Lower>().solve(b);
    }

    inline Eigen::MatrixXd mdivide_left_tri_low(const Eigen::MatrixXd &A) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      int n = A.rows();
      Eigen::MatrixXd b;
      b.setIdentity(n,n);
      A.triangularView<Eigen::Lower>().solveInPlace(b);
      return b;
    }

    /**
     * Returns the solution of the system Ax=b when A is triangular
     * @param A Triangular matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_left_tri(const Eigen::Matrix<double,R1,C1> &A,
                                                        const Eigen::Matrix<double,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      if (A.cols() != b.rows())
        throw std::domain_error("A.cols() != b.rows()");
      return A.template triangularView<TriView>().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular and b=I.
     * @param A Triangular matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @return x = A^-1 .
     * @throws std::domain_error if A is not square
     */
    template<int TriView>
    inline Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mdivide_left_tri(const Eigen::MatrixXd &A) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      int n = A.rows();
      Eigen::MatrixXd b;
      b.setIdentity(n,n);
      A.triangularView<TriView>().solveInPlace(b);
      return b;
    }
    /**
     * Returns the solution of the system Ax=b when A is triangular
     * @param A Triangular matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int TriView,int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_right_tri(const Eigen::Matrix<double,R1,C1> &b,
                                                         const Eigen::Matrix<double,R2,C2> &A) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      if (A.rows() != b.cols())
        throw std::domain_error("A.rows() != b.cols()");
      return A.template triangularView<TriView>().transpose().solve(b.transpose()).transpose();
    }
    /**
     * Returns the solution of the system tri(A)x=b when tri(A) is a
     * lower triangular view of the matrix A.
     * @param A Matrix.
     * @param b Right hand side matrix or vector.
     * @return x = tri(A)^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_right_tri_low(const Eigen::Matrix<double,R1,C1> &b,
                                                             const Eigen::Matrix<double,R2,C2> &A) {
      return mdivide_right_tri<Eigen::Lower>(b,A);
    }


    /**
     * Returns the solution of the system Ax=b.
     * @param A Matrix.
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_left(const Eigen::Matrix<double,R1,C1> &A,
                                                    const Eigen::Matrix<double,R2,C2> &b) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      if (A.cols() != b.rows())
        throw std::domain_error("A.cols() != b.rows()");
      return A.lu().solve(b);
    }
    /**
     * Returns the solution of the system Ax=b.
     * @param A Matrix.
     * @param b Right hand side matrix or vector.
     * @return x = A^-1 b, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2> mdivide_right(const Eigen::Matrix<double,R1,C1> &b,
                                                     const Eigen::Matrix<double,R2,C2> &A) {
      if (A.cols() != A.rows())
        throw std::domain_error("A is not square");
      if (A.rows() != b.cols())
        throw std::domain_error("A.rows() != b.cols()");
      return A.transpose().lu().solve(b.transpose()).transpose();
    }

    /**
     * Return the real component of the eigenvalues of the specified
     * matrix in descending order of magnitude.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Specified matrix.
     * @return Eigenvalues of matrix.
     */
    vector_d eigenvalues(const matrix_d& m);

    /**
     * Return a matrix whose columns are the real components of the
     * eigenvectors of the specified matrix.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Specified matrix.
     * @return Eigenvectors of matrix.
     */
    matrix_d eigenvectors(const matrix_d& m);
    /**
     * Assign the real components of the eigenvalues and eigenvectors
     * of the specified matrix to the specified references.
     * <p>Given an input matrix \f$A\f$, the
     * eigenvalues will be found in \f$D\f$ in descending order of
     * magnitude.  The eigenvectors will be written into
     * the columns of \f$V\f$.  If \f$A\f$ is invertible, then
     * <p>\f$A = V \times \mbox{\rm diag}(D) \times V^{-1}\f$, where
     $ \f$\mbox{\rm diag}(D)\f$ is the square diagonal matrix with
     * diagonal elements \f$D\f$ and \f$V^{-1}\f$ is the inverse of
     * \f$V\f$.
     * @param m Specified matrix.
     * @param eigenvalues Column vector reference into which
     * eigenvalues are written.
     * @param eigenvectors Matrix reference into which eigenvectors
     * are written.
     */
    void eigen_decompose(const matrix_d& m,
                         vector_d& eigenvalues,
                         matrix_d& eigenvectors);

    /**
     * Return the eigenvalues of the specified symmetric matrix
     * in descending order of magnitude.  This function is more
     * efficient than the general eigenvalues function for symmetric
     * matrices.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Specified matrix.
     * @return Eigenvalues of matrix.
     */
    vector_d eigenvalues_sym(const matrix_d& m);
    /**
     * Return a matrix whose rows are the real components of the
     * eigenvectors of the specified symmetric matrix.  This function
     * is more efficient than the general eigenvectors function for
     * symmetric matrices.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Symmetric matrix.
     * @return Eigenvectors of matrix.
     */
    matrix_d eigenvectors_sym(const matrix_d& m);
    /**
     * Assign the real components of the eigenvalues and eigenvectors
     * of the specified symmetric matrix to the specified references.
     * <p>See <code>eigen_decompose()</code> for more information on the
     * values.
     * @param m Symmetric matrix.  This function is more efficient
     * than the general decomposition method for symmetric matrices.
     * @param eigenvalues Column vector reference into which
     * eigenvalues are written.
     * @param eigenvectors Matrix reference into which eigenvectors
     * are written.
     */
    void eigen_decompose_sym(const matrix_d& m,
                                    vector_d& eigenvalues,
                                    matrix_d& eigenvectors);


    /**
     * Return the lower-triangular Cholesky factor (i.e., matrix
     * square root) of the specified square, symmetric matrix.  The return
     * value \f$L\f$ will be a lower-traingular matrix such that the
     * original matrix \f$A\f$ is given by
     * <p>\f$A = L \times L^T\f$.
     * @param m Symmetrix matrix.
     * @return Square root of matrix.
     * @throw std::domain_error if m is not a square matrix
     */
    matrix_d cholesky_decompose(const matrix_d& m);

    /**
     * Return the vector of the singular values of the specified matrix
     * in decreasing order of magnitude.
     * <p>See the documentation for <code>svd()</code> for
     * information on the signular values.
     * @param m Specified matrix.
     * @return Singular values of the matrix.
     */
    vector_d singular_values(const matrix_d& m);

    /**
     * Assign the real components of a singular value decomposition
     * of the specified matrix to the specified references.  
     * <p>Thesingular values \f$S\f$ are assigned to a vector in 
     * decreasing order of magnitude.  The left singular vectors are
     * found in the columns of \f$U\f$ and the right singular vectors
     * in the columns of \f$V\f$.
     * <p>The original matrix is recoverable as
     * <p>\f$A = U \times \mbox{\rm diag}(S) \times V^T\f$, where
     * \f$\mbox{\rm diag}(S)\f$ is the square diagonal matrix with
     * diagonal elements \f$S\f$.
     * <p>If \f$A\f$ is an \f$M \times N\f$ matrix
     * and \f$K = \mbox{\rm min}(M,N)\f$, 
     * then \f$U\f$ is an \f$M \times K\f$ matrix,  
     * \f$S\f$ is a length \f$K\f$ column vector, and 
     * \f$V\f$ is an \f$N \times K\f$ matrix.
     * @param m Matrix to decompose.
     * @param u Left singular vectors.
     * @param v Right singular vectors.
     * @param s Singular values.
     */
    void svd(const matrix_d& m, matrix_d& u, matrix_d& v, vector_d& s);


    template <typename T>
    inline 
    std::vector<T>
    cumulative_sum(const std::vector<T>& x) {
      std::vector<T> result(x.size());
      if (x.size() == 0)
        return result;
      result[0] = x[0];
      for (size_t i = 1; i < result.size(); ++i)
        result[i] = x[i] + result[i-1];
      return result;
    }
    template <typename T, int C>
    inline 
    Eigen::Matrix<T,1,C> 
    cumulative_sum(const Eigen::Matrix<T,1,C>& rv) {
      Eigen::Matrix<T,1,C> result(rv.size());
      if (rv.size() == 0)
        return result;
      result(0) = rv(0);
      for (int i = 1; i < result.size(); ++i)
        result(i) = rv(i) + result(i-1);
      return result;
    }
    template <typename T, int R>
    inline 
    Eigen::Matrix<T,R,1>
    cumulative_sum(const Eigen::Matrix<T,R,1>& v) {
      Eigen::Matrix<T,R,1> result(v.size());
      if (v.size() == 0)
        return result;
      result(0) = v(0);
      for (int i = 1; i < result.size(); ++i)
        result(i) = v(i) + result(i-1);
      return result;
    }


    template <typename T>
    void stan_print(std::ostream* o, const T& x) {
      *o << x;
    }
    
    template <typename T>
    void stan_print(std::ostream* o, const std::vector<T>& x) {
      *o << '[';
      for (int i = 0; i < x.size(); ++i) {
        if (i > 0) *o << ',';
        stan_print(o,x[i]);
      }
      *o << ']';
    }

    template <typename T>
    void stan_print(std::ostream* o, const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      *o << '[';
      for (int i = 0; i < x.size(); ++i) {
        if (i > 0) *o << ',';
        stan_print(o,x(i));
      }
      *o << ']';
    }

    template <typename T>
    void stan_print(std::ostream* o, const Eigen::Matrix<T,1,Eigen::Dynamic>& x) {
      *o << '[';
      for (int i = 0; i < x.size(); ++i) {
        if (i > 0) *o << ',';
        stan_print(o,x(i));
      }
      *o << ']';
    }

    template <typename T>
    void stan_print(std::ostream* o, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x) {
      *o << '[';
      for (int i = 0; i < x.rows(); ++i) {
        if (i > 0) *o << ',';
        stan_print(o,x.row(i));
      }
      *o << ']';
    }


  }

}

#endif

