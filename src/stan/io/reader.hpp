#ifndef __STAN__IO__READER_HPP__
#define __STAN__IO__READER_HPP__

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <vector>

#include <boost/throw_exception.hpp>

#include <stan/math/error_handling.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/special_functions.hpp>

#include <stan/prob/transform.hpp>

namespace stan {

  namespace io {


    /**
     * A stream-based reader for integer, scalar, vector, matrix
     * and array data types, with Jacobian calculations.
     *
     * The template parameter <code>T</code> represents the type of
     * scalars and the values in vectors and matrices.  The only
     * requirement on the template type <code>T</code> is that a
     * double can be copied into it, as in
     *
     * <code>T t = 0.0;</code>
     *
     * This includes <code>double</code> itself and the reverse-mode
     * algorithmic differentiation class <code>stan::agrad::var</code>.
     *
     * <p>For transformed values, the scalar type parameter <code>T</code>
     * must support the transforming operations, such as <code>exp(x)</code>
     * for positive-bounded variables.  It must also support equality and
     * inequality tests with <code>double</code> values.
     *
     * @tparam T Basic scalar type.
     */
    template <typename T>
    class reader {

    private:

      std::vector<T>& data_r_;
      std::vector<int>& data_i_;
      size_t pos_;
      size_t int_pos_;
      
      inline T& scalar_ptr() {
        return data_r_.at(pos_);
      }
      
      inline T& scalar_ptr_increment(size_t m) {
        pos_ += m;
        return data_r_.at(pos_ - m);
      }

      inline int& int_ptr() {
        return data_i_.at(int_pos_);
      }
      
      inline int& int_ptr_increment(size_t m) {
        int_pos_ += m;
        return data_i_.at(int_pos_ - m);
      }

    public:

      typedef typename stan::math::EigenType<T>::matrix matrix_t;
      typedef typename stan::math::EigenType<T>::vector vector_t;
      typedef typename stan::math::EigenType<T>::row_vector row_vector_t;

      typedef Eigen::Map<matrix_t> map_matrix_t;
      typedef Eigen::Map<vector_t> map_vector_t;
      typedef Eigen::Map<row_vector_t> map_row_vector_t;


      /**
       * Construct a variable reader using the specified vectors
       * as the source of scalar and integer values for data.  This
       * class holds a reference to the specified data vectors.
       *
       * Attempting to read beyond the end of the data or integer
       * value sequences raises a runtime exception.
       *
       * @param data_r Sequence of scalar values.
       * @param data_i Sequence of integer values.
       */
      reader(std::vector<T>& data_r,
             std::vector<int>& data_i) 
        : data_r_(data_r),
          data_i_(data_i),
          pos_(0),
          int_pos_(0) {
      }

      /**
       * Destroy this variable reader.  
       */
      ~reader() { }

      /**
       * Return the number of scalars remaining to be read.
       *
       * @return Number of scalars left to read.
       */
      inline size_t available() {
        return data_r_.size() - pos_;
      }

      /**
       * Return the number of integers remaining to be read.
       *
       * @return Number of integers left to read.
       */
      inline size_t available_i() {
        return data_i_.size() - int_pos_;
      }

      /**
       * Return the next integer in the integer sequence.
       *
       * @return Next integer value.
       */
      inline int integer() {
        if (int_pos_ >= data_i_.size())
          BOOST_THROW_EXCEPTION(
              std::runtime_error("no more integers to read."));
        return data_i_[int_pos_++];
      }

      /**
       * Return the next integer in the integer sequence.
       * This form is a convenience method to make compiling
       * easier; its behavior is the same as <code>int()</code>
       *
       * @return Next integer value.
       */
      inline int integer_constrain() {
        return integer();
      }
      
      /**
       * Return the next integer in the integer sequence.
       * This form is a convenience method to make compiling
       * easier; its behavior is the same as <code>integer()</code>
       *
       * @return Next integer value.
       */
      inline int integer_constrain(T& log_prob) {
        return integer();
      }
      


      /**
       * Return the next scalar in the sequence.
       *
       * @return Next scalar value.
       */
      inline T scalar() {
        if (pos_ >= data_r_.size())
          BOOST_THROW_EXCEPTION(std::runtime_error("no more scalars to read"));
        return data_r_[pos_++];
      }

      /**
       * Return the next scalar.  For arbitrary scalars,
       * constraint is a no-op.
       *
       * @return Next scalar.
       */
      inline T scalar_constrain() {
        return scalar();
      }

      /**
       * Return the next scalar in the sequence, incrementing
       * the specified reference with the log absolute Jacobian determinant.
       * 
       * <p>With no transformation, the Jacobian increment is a no-op.
       * 
       * <p>See <code>scalar_constrain()</code>.  
       *
       * @param log_prob Reference to log probability variable to increment.
       * @return Next scalar.
       */
      T scalar_constrain(T& log_prob) {
        return scalar();
      }


      /**
       * Return a standard library vector of the specified
       * dimensionality made up of the next scalars.
       *
       * @param m Size of vector.
       * @return Vector made up of the next scalars.
       */
      inline std::vector<T> std_vector(size_t m) {
        std::vector<T> vec;
        T& start = scalar_ptr_increment(m);
        vec.insert(vec.begin(), &start, &scalar_ptr());
        return vec;
      }

      /**
       * Return a column vector of specified dimensionality made up of
       * the next scalars.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      inline vector_t vector(size_t m) {
        return map_vector_t(&scalar_ptr_increment(m),m);
      }

      // FIXME:  replace remaining Eigen::Matrix w. EigenType


      /**
       * Return a column vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      inline vector_t vector_constrain(size_t m) {
        return map_vector_t(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a column vector of specified dimensionality made up of
       * the next scalars.  The constraint and hence Jacobian are no-ops.
       *
       * @param m Number of rows in the vector to read.
       * @param lp Log probability to increment.
       * @return Column vector made up of the next scalars.
       */
      inline vector_t vector_constrain(size_t m, T& lp) {
        return map_vector_t(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      inline row_vector_t row_vector(size_t m) {
        return map_row_vector_t(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      inline row_vector_t row_vector_constrain(size_t m) {
        return map_row_vector_t(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op, so the log
       * probability is not incremented.
       *
       * @param m Number of rows in the vector to read.
       * @param lp Log probability to increment.
       * @return Column vector made up of the next scalars.
       */
      inline row_vector_t row_vector_constrain(size_t m, T& lp) {
        return map_row_vector_t(&scalar_ptr_increment(m),m);
      }
      
      /**
       * Return a matrix of the specified dimensionality made up of
       * the next scalars arranged in column-major order.
       *
       * Row-major reading means that if a matrix of <code>m=2</code>
       * rows and <code>n=3</code> columns is reada and the next
       * scalar values are <code>1,2,3,4,5,6</code>, the result is 
       *
       * <pre> 
       * a = 1 4
       *     2 5
       *     3 6</pre>
       *
       * @param m Number of rows.  
       * @param n Number of columns.
       * @return Eigen::Matrix made up of the next scalars.
       */
      inline matrix_t matrix(size_t m, size_t n) {
        return map_matrix_t(&scalar_ptr_increment(m*n),m,n);
      }

      /**
       * Return a matrix of the specified dimensionality made up of
       * the next scalars arranged in column-major order.  The
       * constraint is a no-op.  See <code>matrix(size_t,
       * size_t)</code> for more information.
       *
       * @param m Number of rows.  
       * @param n Number of columns.
       * @return Matrix made up of the next scalars.
       */
      inline matrix_t matrix_constrain(size_t m, size_t n) {
        return map_matrix_t(&scalar_ptr_increment(m*n),m,n);
      }

      /**
       * Return a matrix of the specified dimensionality made up of
       * the next scalars arranged in column-major order.  The
       * constraint is a no-op, hence the log probability is not
       * incremented.  See <code>matrix(size_t, size_t)</code>
       * for more information.
       *
       * @param m Number of rows.  
       * @param n Number of columns.
       * @param lp Log probability to increment.
       * @return Matrix made up of the next scalars.
       */
      inline matrix_t matrix_constrain(size_t m, size_t n, T& lp) {
        return map_matrix_t(&scalar_ptr_increment(m*n),m,n);
      }


      /**
       * Return the next integer, checking that it is greater than
       * or equal to the specified lower bound.
       *
       * @param lb Lower bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * greater than or equal to the lower bound.
       */
      inline int integer_lb(int lb) {
        int i = integer();
        if (!(i >= lb))
          BOOST_THROW_EXCEPTION(
              std::runtime_error("required value greater than or equal to lb"));
        return i;
      }
      /**
       * Return the next integer, checking that it is greater than
       * or equal to the specified lower bound.
       * 
       * @param lb Lower bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * greater than or equal to the lower bound.
       */
      inline int integer_lb_constrain(int lb) {
        return integer_lb(lb);
      }
      /**
       * Return the next integer, checking that it is greater than
       * or equal to the specified lower bound.
       * 
       * @param lb Lower bound.
       * @param lp Log probability (ignored because no Jacobian)
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * greater than or equal to the lower bound.
       */
      inline int integer_lb_constrain(int lb, T& lp) {
        return integer_lb(lb);
      }


      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.
       *
       * @param ub Upper bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      inline int integer_ub(int ub) {
        int i = integer();
        if (!(i <= ub))
          BOOST_THROW_EXCEPTION(
              std::runtime_error("required value less than or equal to ub"));
        return i;
      }
      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.
       * 
       * @param ub Upper bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      inline int integer_ub_constrain(int ub) {
        return integer_ub(ub);
      }
      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.
       * 
       * @param ub Upper bound.
       * @param lp Log probability (ignored because no Jacobian)
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      int integer_ub_constrain(int ub, T& lp) {
        return integer_ub(ub);
      }

      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.  Even if the upper
       * bounds and lower bounds are not consistent, the next integer
       * value will be consumed.
       *
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      inline int integer_lub(int lb, int ub) {
        // read first to make position deterministic [arbitrary choice]
        int i = integer(); 
        if (lb > ub)
          BOOST_THROW_EXCEPTION(
            std::runtime_error("lower bound must be less than or equal to ub"));
        if (!(i >= lb))
          BOOST_THROW_EXCEPTION(
            std::runtime_error("required value greater than or equal to lb"));
        if (!(i <= ub))
          BOOST_THROW_EXCEPTION(
            std::runtime_error("required value less than or equal to ub"));
        return i;
      }
      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.
       * 
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      inline int integer_lub_constrain(int lb, int ub) {
        return integer_lub(lb,ub);
      }
      /**
       * Return the next integer, checking that it is less than
       * or equal to the specified upper bound.
       * 
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @param lp Log probability (ignored because no Jacobian)
       * @return Next integer read.
       * @throw std::runtime_error If the next integer read is not
       * less than or equal to the upper bound.
       */
      inline int integer_lub_constrain(int lb, int ub, T& lp) {
        return integer_lub(lb,ub);
      }
      


      /**
       * Return the next scalar, checking that it is
       * positive.  
       *
       * <p>See <code>stan::math::check_positive(T)</code>.
       *
       * @return Next positive scalar.
       * @throw std::runtime_error if x is not positive
       */
      inline T scalar_pos() {
        T x(scalar());
        stan::math::check_positive("stan::io::scalar_pos(%1%)", x, "x");
        return x;
      }

      /**
       * Return the next scalar, transformed to be positive.
       *
       * <p>See <code>stan::prob::positive_constrain(T)</code>.
       *
       * @return The next scalar transformed to be positive.
       */
      inline T scalar_pos_constrain() {
        return stan::prob::positive_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be positive,
       * incrementing the specified reference with the log absolute
       * determinant of the Jacobian.
       *
       * <p>See <code>stan::prob::positive_constrain(T,T&)</code>.
       * 
       * @param lp Reference to log probability variable to increment.
       * @return The next scalar transformed to be positive.
       */
      inline T scalar_pos_constrain(T& lp) {
        return stan::prob::positive_constrain(scalar(),lp);
      }

      /**
       * Return the next scalar, checking that it is
       * greater than or equal to the specified lower bound.
       *
       * <p>See <code>stan::math::check_greater_or_equal(T,double)</code>.
       *
       * @param lb Lower bound.
       * @return Next scalar value.
       * @tparam TL Type of lower bound.
       * @throw std::runtime_error if the scalar is less than the
       *    specified lower bound
       */
      template <typename TL>
      inline T scalar_lb(const TL lb) {
        T x(scalar());
        stan::math::check_greater_or_equal("stan::io::scalar_lb(%1%)",
                                           x, lb, "x");
        return x;
      }

      /**
       * Return the next scalar transformed to have the
       * specified lower bound.
       *
       * <p>See <code>stan::prob::lb_constrain(T,double)</code>.
       *
       * @tparam TL Type of lower bound.
       * @param lb Lower bound on values.
       * @return Next scalar transformed to have the specified
       * lower bound.
       */
      template <typename TL>
      inline T scalar_lb_constrain(const TL lb) {
        return stan::prob::lb_constrain(scalar(),lb);
      }

      /**
       * Return the next scalar transformed to have the specified
       * lower bound, incrementing the specified reference with the
       * log of the absolute Jacobian determinant of the transform.
       *
       * <p>See <code>stan::prob::lb_constrain(T,double,T&)</code>.
       *
       * @tparam TL Type of lower bound.
       * @param lb Lower bound on result.
       * @param lp Reference to log probability variable to increment.
       */
      template <typename TL>
      inline T scalar_lb_constrain(const TL lb, T& lp) {
        return stan::prob::lb_constrain(scalar(),lb,lp);
      }



      /**
       * Return the next scalar, checking that it is
       * less than or equal to the specified upper bound.
       *
       * <p>See <code>stan::math::check_less_or_equal(T,double)</code>.
       *
       * @tparam TU Type of upper bound.
       * @param ub Upper bound.
       * @return Next scalar value.
       * @throw std::runtime_error if the scalar is greater than the
       *    specified upper bound
       */
      template <typename TU>
      inline T scalar_ub(TU ub) {
        T x(scalar());
        stan::math::check_less_or_equal("stan::io::scalar_ub(%1%)", x, ub, "x");
        return x;
      }

      /**
       * Return the next scalar transformed to have the
       * specified upper bound.
       *
       * <p>See <code>stan::prob::ub_constrain(T,double)</code>.
       *
       * @tparam TU Type of upper bound.
       * @param ub Upper bound on values.
       * @return Next scalar transformed to have the specified
       * upper bound.
       */
      template <typename TU>
      inline T scalar_ub_constrain(const TU ub) {
        return stan::prob::ub_constrain(scalar(),ub);
      }

      /**
       * Return the next scalar transformed to have the specified
       * upper bound, incrementing the specified reference with the
       * log of the absolute Jacobian determinant of the transform.
       *
       * <p>See <code>stan::prob::ub_constrain(T,double,T&)</code>.
       *
       * @tparam TU Type of upper bound.
       * @param ub Upper bound on result.
       * @param lp Reference to log probability variable to increment.
       */
      template <typename TU>
      inline T scalar_ub_constrain(const TU ub, T& lp) {
        return stan::prob::ub_constrain(scalar(),ub,lp);
      }

      /**
       * Return the next scalar, checking that it is between
       * the specified lower and upper bound.
       *
       * <p>See <code>stan::math::check_bounded(T,double,double)</code>.
       *
       * @tparam TL Type of lower bound.
       * @tparam TU Type of upper bound.
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next scalar value.
       * @throw std::runtime_error if the scalar is not between the specified
       *    lower and upper bounds.
       */
      template <typename TL, typename TU>
      inline T scalar_lub(const TL lb, const TU ub) {
        T x(scalar());
        stan::math::check_bounded("stan::io::scalar_lub(%1%)", x, lb, ub, "x");
        return x;
      }

      /**
       * Return the next scalar transformed to be between
       * the specified lower and upper bounds.
       *
       * <p>See <code>stan::prob::lub_constrain(T,double,double)</code>.
       *
       * @tparam TL Type of lower bound.
       * @tparam TU Type of upper bound.
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next scalar transformed to fall between the specified
       * bounds.
       */
      template <typename TL, typename TU>
      inline T scalar_lub_constrain(const TL lb, const TU ub) {
        return stan::prob::lub_constrain(scalar(),lb,ub);
      }

      /**
       * Return the next scalar transformed to be between the 
       * the specified lower and upper bounds.
       * 
       * <p>See <code>stan::prob::lub_constrain(T,double,double,T&)</code>.
       * 
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @param lp Reference to log probability variable to increment.
       * @tparam T Type of scalar.
       * @tparam TL Type of lower bound.
       * @tparam TU Type of upper bound.
       */
      template <typename TL, typename TU>
      inline T scalar_lub_constrain(TL lb, TU ub, T& lp) {
        return stan::prob::lub_constrain(scalar(),lb,ub,lp);
      }

      /**
       * Return the next scalar, checking that it is a valid value for
       * a probability, between 0 (inclusive) and 1 (inclusive).
       *
       * <p>See <code>stan::math::check_bounded(T)</code>.
       * 
       * @return Next probability value.
       */
      inline T prob() {
        T x(scalar());
        stan::math::check_bounded("stan::io::prob(%1%)", x, 0, 1, "x");
        return x;
      }

      /**
       * Return the next scalar transformed to be a probability
       * between 0 and 1.
       *
       * <p>See <code>stan::prob::prob_constrain(T)</code>.
       *
       * @return The next scalar transformed to a probability.
       */
      inline T prob_constrain() {
        return stan::prob::prob_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be a probability
       * between 0 and 1, incrementing the specified reference with
       * the log of the absolute Jacobian determinant.
       * 
       * <p>See <code>stan::prob::prob_constrain(T)</code>.
       *
       * @param lp Reference to log probability variable to increment.
       * @return The next scalar transformed to a probability.
       */
      inline T prob_constrain(T& lp) {
        return stan::prob::prob_constrain(scalar(),lp);
      }




      /**
       * Return the next scalar, checking that it is a valid
       * value for a correlation, between -1 (inclusive) and
       * 1 (inclusive).
       *
       * <p>See <code>stan::math::check_bounded(T)</code>.
       *
       * @return Next correlation value.
       * @throw std::runtime_error if the value is not valid
       *   for a correlation
       */
      inline T corr() {
        T x(scalar());
        stan::math::check_bounded("stan::io::corr(%1%)", x, -1, 1, "x");
        return x;
      }

      /**
       * Return the next scalar transformed to be a correlation
       * between -1 and 1.
       *
       * <p>See <code>stan::prob::corr_constrain(T)</code>.
       *
       * @return The next scalar transformed to a correlation.
       */
      inline T corr_constrain() {
        return stan::prob::corr_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be a (partial)
       * correlation between -1 and 1, incrementing the specified
       * reference with the log of the absolute Jacobian determinant.
       *
       * <p>See <code>stan::prob::corr_constrain(T,T&)</code>.
       * 
       * @param lp The reference to the variable holding the log
       * probability to increment.
       * @return The next scalar transformed to a correlation.
       */
      inline T corr_constrain(T& lp) {
        return stan::prob::corr_constrain(scalar(),lp);
      }

      /**
       * Return a simplex of the specified size made up of the
       * next scalars.  
       *
       * <p>See <code>stan::math::check_simplex</code>.
       *
       * @param k Size of returned simplex.
       * @return Simplex read from the specified size number of scalars.
       * @throw std::runtime_error if the k values is not a simplex.
       */
      inline vector_t simplex(size_t k) {
        vector_t theta(vector(k));
        stan::math::check_simplex("stan::io::simplex(%1%)", theta, "theta");
        return theta;
      }

      /**
       * Return the next simplex transformed vector of the specified
       * length.  This operation consumes one less than the specified
       * length number of scalars.  
       *
       * <p>See <code>stan::prob::simplex_constrain(Eigen::Matrix)</code>.
       *
       * @param k Number of dimensions in resulting simplex.
       * @return Simplex derived from next <code>k-1</code> scalars.
       */
      inline 
      Eigen::Matrix<T,Eigen::Dynamic,1> simplex_constrain(size_t k) {
        return stan::prob::simplex_constrain(vector(k-1));
      }

      /**
       * Return the next simplex of the specified size (using one fewer
       * unconstrained scalars), incrementing the specified reference with the
       * log absolute Jacobian determinant.
       *
       * <p>See <code>stan::prob::simplex_constrain(Eigen::Matrix,T&)</code>.
       *
       * @param k Size of simplex.
       * @param lp Log probability to increment with log absolute
       * Jacobian determinant.
       * @return The next simplex of the specified size.
       */
      inline vector_t simplex_constrain(size_t k, T& lp) {
        return stan::prob::simplex_constrain(vector(k-1),lp);
      }

      /**
       * Return the next vector of specified size containing positive
       * values in order.  
       *
       * <p>See <code>stan::math::check_ordered(T)</code>.
       *
       * @param k Size of returned vector.
       * @return Vector of positive values in ascending order.
       * @throw std::runtime_error if the vector is not positive ordered
       */
      inline vector_t ordered(size_t k) {
        vector_t x(vector(k));
        stan::math::check_ordered("stan::io::ordered(%1%)", x, "x");
        return x;
      }

      /**
       * Return the next positive, ordered vector of the specified
       * length.  
       *
       * <p>See <code>stan::prob::ordered_constrain(Matrix)</code>.
       * 
       * @param k Length of returned vector.
       * @return Next positive, ordered vector of the specified
       * length.
       */
      inline vector_t ordered_constrain(size_t k) {
        return stan::prob::ordered_constrain(vector(k));
      }

      /**
       * Return the next positive ordered vector of the specified
       * size, incrementing the specified reference with the log
       * absolute Jacobian of the determinant.
       *
       * <p>See <code>stan::prob::ordered_constrain(Matrix,T&)</code>.
       *
       * @param k Size of vector.
       * @param lp Log probability reference to increment.
       * @return Next positive ordered vector of the specified size.
       */
      inline vector_t ordered_constrain(size_t k, T& lp) {
        return stan::prob::ordered_constrain(vector(k),lp);
      }

      /**
       * Returns the next correlation matrix of the specified dimensionality.
       *
       * <p>See <code>stan::math::check_corr_matrix(Matrix)</code>.
       *
       * @param k Dimensionality of correlation matrix.
       * @return Next correlation matrix of the specified dimensionality.
       * @throw std::runtime_error if the matrix is not a correlation matrix
       */
      inline matrix_t corr_matrix(size_t k) {
        matrix_t x(matrix(k,k));
        stan::math::check_corr_matrix("stan::math::corr_matrix(%1%)", x, "x");
        return x;
      }

      /**
       * Return the next correlation matrix of the specified dimensionality.
       *
       * <p>See <code>stan::prob::corr_matrix_constrain(Matrix)</code>.
       *
       * @param k Dimensionality of correlation matrix.
       * @return Next correlation matrix of the specified dimensionality.
       */
      inline matrix_t corr_matrix_constrain(size_t k) {
        return stan::prob::corr_matrix_constrain(vector((k * (k - 1)) / 2),k);
      }

      /**
       * Return the next correlation matrix of the specified dimensionality,
       * incrementing the specified reference with the log absolute Jacobian
       * determinant.
       * 
       * <p>See <code>stan::prob::corr_matrix_constrain(Matrix,T&)</code>.
       *
       * @param k Dimensionality of the (square) correlation matrix.
       * @param lp Log probability reference to increment.
       * @return The next correlation matrix of the specified dimensionality.
       */
      inline matrix_t corr_matrix_constrain(size_t k, T& lp) {
        return stan::prob::corr_matrix_constrain(vector((k * (k - 1)) / 2),
                                                 k,lp);
      }


      /**
       * Return the next covariance matrix with the specified 
       * dimensionality.  
       *
       * <p>See <code>stan::math::check_cov_matrix(Matrix)</code>.
       *
       * @param k Dimensionality of covariance matrix.
       * @return Next covariance matrix of the specified dimensionality.
       * @throw std::runtime_error if the matrix is not a valid
       *    covariance matrix
       */
      inline matrix_t cov_matrix(size_t k) {
        matrix_t y(matrix(k,k));
        stan::math::check_cov_matrix("stan::io::cov_matrix(%1%)", y, "y");
        return y;
      }

      /**
       * Return the next covariance matrix of the specified dimensionality.
       *
       * <p>See <code>stan::prob::cov_matrix_constrain(Matrix)</code>.
       * 
       * @param k Dimensionality of covariance matrix.
       * @return Next covariance matrix of the specified dimensionality.
       */
      inline matrix_t cov_matrix_constrain(size_t k) {
        return stan::prob::cov_matrix_constrain(vector(k + (k * (k - 1)) / 2),
                                                k);
      }

      /**
       * Return the next covariance matrix of the specified dimensionality,
       * incrementing the specified reference with the log absolute Jacobian
       * determinant.
       * 
       * <p>See <code>stan::prob::cov_matrix_constrain(Matrix,T&)</code>.
       *
       * @param k Dimensionality of the (square) covariance matrix.
       * @param lp Log probability reference to increment.
       * @return The next covariance matrix of the specified dimensionality.
       */
      inline matrix_t cov_matrix_constrain(size_t k, T& lp) {
        return stan::prob::cov_matrix_constrain(vector(k + (k * (k - 1)) / 2),
                                                k,lp);
      }


    };

  }

}

#endif
