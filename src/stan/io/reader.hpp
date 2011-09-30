#ifndef __STAN__IO__READER_H__
#define __STAN__IO__READER_H__

#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <stan/maths/special_functions.hpp>
#include <stan/prob/transform.hpp>

namespace stan {

  namespace io {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    

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
      unsigned int pos_;
      unsigned int int_pos_;
      
      inline T& scalar_ptr() {
	return data_r_.at(pos_);
      }
      
      inline T& scalar_ptr_increment(unsigned int m) {
	pos_ += m;
	return data_r_.at(pos_ - m);
      }

      inline int& int_ptr() {
	return data_i_.at(int_pos_);
      }
      
      inline int& int_ptr_increment(unsigned int m) {
	int_pos_ += m;
	return data_i_.at(int_pos_ - m);
      }

    public:

      /**
       * Construct a variable reader using the specified vectors
       * as the source of scalar and integer values for data.  This
       * class holds a reference to the specified data vectors.
       *
       * @param data_r Sequence of scalar values.
       * @param data_i Sequence of integer values.
       */
      reader(std::vector<T>& data_r,
	     std::vector<int>& data_i) 
	: data_r_(data_r),
	  data_i_(data_i),
	  pos_(0) {
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
      inline unsigned int available() {
	return data_r_.size() - pos_;
      }

      /**
       * Return the next integer in the integer sequence.
       *
       * @return Next integer value.
       */
      inline int integer() {
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
	return data_i_[int_pos_++];
      }
      
      /**
       * Return the next integer in the integer sequence.
       * This form is a convenience method to make compiling
       * easier; its behavior is the same as <code>integer()</code>
       *
       * @return Next integer value.
       */
      inline int integer_constrain(T& log_prob) {
	return data_i_[int_pos_++];
      }
      


      /**
       * Return the next scalar in the sequence.
       *
       * @return Next scalar value.
       */
      inline T scalar() {
	return data_r_[pos_++];
      }

      /**
       * Return the next scalar.  For arbitrary scalars,
       * constraint is a no-op.
       *
       * @return Next scalar.
       */
      T scalar_constrain() {
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
      std::vector<T> std_vector(unsigned int m) {
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
      Matrix<T,Dynamic,1> vector(unsigned int m) {
	return Eigen::Map<Matrix<T,Dynamic,1> >(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a column vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      Matrix<T,Dynamic,1> vector_constrain(unsigned int m) {
	return Eigen::Map<Matrix<T,Dynamic,1> >(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a column vector of specified dimensionality made up of
       * the next scalars.  The constraint and hence Jacobian are no-ops.
       *
       * @param m Number of rows in the vector to read.
       * @param lp Log probability to increment.
       * @return Column vector made up of the next scalars.
       */
      Matrix<T,Dynamic,1> vector_constrain(unsigned int m, T& lp) {
	return Eigen::Map<Matrix<T,Dynamic,1> >(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      Matrix<T,Dynamic,1> row_vector(unsigned int m) {
	return Eigen::Map<Matrix<T,1,Dynamic> >(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      Matrix<T,Dynamic,1> row_vector_constrain(unsigned int m) {
	return Eigen::Map<Matrix<T,1,Dynamic> >(&scalar_ptr_increment(m),m);
      }

      /**
       * Return a row vector of specified dimensionality made up of
       * the next scalars.  The constraint is a no-op, so the log
       * probability is not incremented.
       *
       * @param m Number of rows in the vector to read.
       * @return Column vector made up of the next scalars.
       */
      Matrix<T,Dynamic,1> row_vector_constrain(unsigned int m, T& lp) {
	return Eigen::Map<Matrix<T,1,Dynamic> >(&scalar_ptr_increment(m),m);
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
       * @return Matrix made up of the next scalars.
       */
      Matrix<T,Dynamic,Dynamic> matrix(unsigned int m, unsigned int n) {
	return Eigen::Map<Matrix<T,Dynamic,Dynamic> >(&scalar_ptr_increment(m*n),m,n);
      }

      /**
       * Return a matrix of the specified dimensionality made up of
       * the next scalars arranged in column-major order.  The
       * constraint is a no-op.  See <code>matrix(unsigned int, unsigned int)</code>
       * for more information.
       *
       * @param m Number of rows.  
       * @param n Number of columns.
       * @return Matrix made up of the next scalars.
       */
      Matrix<T,Dynamic,Dynamic> matrix_constrain(unsigned int m, unsigned int n) {
	return Eigen::Map<Matrix<T,Dynamic,Dynamic> >(&scalar_ptr_increment(m*n),m,n);
      }

      /**
       * Return a matrix of the specified dimensionality made up of
       * the next scalars arranged in column-major order.  The
       * constraint is a no-op, hence the log probability is not
       * incremented.  See <code>matrix(unsigned int, unsigned int)</code>
       * for more information.
       *
       * @param m Number of rows.  
       * @param n Number of columns.
       * @param lp Log probability to increment.
       * @return Matrix made up of the next scalars.
       */
      Matrix<T,Dynamic,Dynamic> matrix_constrain(unsigned int m, unsigned int n, T& lp) {
	return Eigen::Map<Matrix<T,Dynamic,Dynamic> >(&scalar_ptr_increment(m*n),m,n);
      }


      /**
       * Return the next scalar, checking that it is
       * positive.  
       *
       * <p>See <code>stan::prob::positive_validate(T)</code>.
       *
       * @return Next positive scalar.
       */
      T scalar_pos() {
	T x(scalar());
	assert(stan::prob::positive_validate(x));
	return x;
      }

      /**
       * Return the next scalar, transformed to be positive.
       *
       * <p>See <code>stan::prob::positive_constrain(T)</code>.
       *
       * @return The next scalar transformed to be positive.
       */
      T scalar_pos_constrain() {
	return stan::prob::positive_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be positive, incrementing
       * the specified reference with the log absolute determinant of the Jacobian.
       *
       * <p>See <code>stan::prob::positive_constrain(T,T&)</code>.
       * 
       * @param lp Reference to log probability variable to increment.
       * @return The next scalar transformed to be positive.
       */
      T scalar_pos_constrain(T& lp) {
	return stan::prob::positive_constrain(scalar(),lp);
      }

      /**
       * Return the next scalar, checking that it is
       * greater than or equal to the specified lower bound.
       *
       * <p>See <code>stan::prob::lb_validate(T,double)</code>.
       *
       * @param lb Lower bound.
       * @return Next scalar value.
       */
      T scalar_lb(double lb) {
	T x(scalar());
	assert(stan::prob::lb_validate(x,lb));
	return x;
      }

      /**
       * Return the next scalar transformed to have the
       * specified lower bound.
       *
       * <p>See <code>stan::prob::lb_constrain(T,double)</code>.
       *
       * @param lb Lower bound on values.
       * @return Next scalar transformed to have the specified
       * lower bound.
       */
      T scalar_lb_constrain(double lb) {
	return stan::prob::lb_constrain(scalar(),lb);
      }

      /**
       * Return the next scalar transformed to have the specified
       * lower bound, incrementing the specified reference with the
       * log of the absolute Jacobian determinant of the transform.
       *
       * <p>See <code>stan::prob::lb_constrain(T,double,T&)</code>.
       *
       * @param lb Lower bound on result.
       * @param lp Reference to log probability variable to increment.
       */
      T scalar_lb_constrain(double lb, T& lp) {
	return stan::prob::lb_constrain(scalar(),lb,lp);
      }



      /**
       * Return the next scalar, checking that it is
       * greater than or equal to the specified lower bound.
       *
       * <p>See <code>stan::prob::ub_validate(T,double)</code>.
       *
       * @param ub Upper bound.
       * @return Next scalar value.
       */
      T scalar_ub(double ub) {
	T x(scalar());
	assert(stan::prob::ub_validate(x,ub));
	return x;
      }

      /**
       * Return the next scalar transformed to have the
       * specified upper bound.
       *
       * <p>See <code>stan::prob::ub_constrain(T,double)</code>.
       *
       * @param ub Upper bound on values.
       * @return Next scalar transformed to have the specified
       * upper bound.
       */
      T scalar_ub_constrain(double ub) {
	return stan::prob::ub_constrain(scalar(),ub);
      }

      /**
       * Return the next scalar transformed to have the specified
       * upper bound, incrementing the specified reference with the
       * log of the absolute Jacobian determinant of the transform.
       *
       * <p>See <code>stan::prob::ub_constrain(T,double,T&)</code>.
       *
       * @param ub Upper bound on result.
       * @param lp Reference to log probability variable to increment.
       */
      T scalar_ub_constrain(double ub, T& lp) {
	return stan::prob::ub_constrain(scalar(),ub,lp);
      }

      /**
       * Return the next scalar, checking that it is between
       * the specified lower and upper bound.
       *
       * <p>See <code>stan::prob::lub_validate(T,double,double)</code>.
       *
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next scalar value.
       */
      T scalar_lub(double lb, double ub) {
	T x(scalar());
	assert(stan::prob::lub_validate(x,lb,ub));
	return x;
      }

      /**
       * Return the next scalar transformed to be between
       * the specified lower and upper bounds.
       *
       * <p>See <code>stan::prob::lub_constrain(T,double,double)</code>.
       *
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @return Next scalar transformed to fall between the specified
       * bounds.
       */
      T scalar_lub_constrain(double lb, double ub) {
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
       */
      T scalar_lub_constrain(double lb, double ub, T& lp) {
	return stan::prob::lub_constrain(scalar(),lb,ub,lp);
      }

      /**
       * Return the next scalar, checking that it is a valid value for
       * a probability, between 0 (inclusive) and 1 (inclusive).
       *
       * <p>See <code>stan::prob::prob_validate(T)</code>.
       * 
       * @return Next probability value.
       */
      T prob() {
	T x(scalar());
	stan::prob::prob_validate(x);
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
      T prob_constrain() {
	return stan::prob::prob_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be a probability between 0 and 1,
       * incrementing the specified reference with the log of the absolute Jacobian
       * determinant.
       * 
       * <p>See <code>stan::prob::prob_constrain(T)</code>.
       *
       * @param lp Reference to log probability variable to increment.
       * @return The next scalar transformed to a probability.
       */
      T prob_constrain(T& lp) {
	return stan::prob::prob_constrain(scalar(),lp);
      }




      /**
       * Return the next scalar, checking that it is a valid
       * value for a correlation, between -1 (inclusive) and
       * 1 (inclusive).
       *
       * <p>See <code>stan::prob::corr_validate(T)</code>.
       *
       * @return Next correlation value.
       */
      T corr() {
	T x(scalar());
	assert(stan::prob::corr_validate(x));
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
      T corr_constrain() {
	return stan::prob::corr_constrain(scalar());
      }

      /**
       * Return the next scalar transformed to be a (partial) correlation between -1 and
       * 1, incrementing the specified reference with the log of the absolute Jacobian
       * determinant.
       *
       * <p>See <code>stan::prob::corr_constrain(T,T&)</code>.
       * 
       * @param log_prob The reference to the variable to increment.
       * @return The next scalar transformed to a correlation.
       */
      T corr_constrain(T& lp) {
	return stan::prob::corr_constrain(scalar(),lp);
      }

      /**
       * Return a simplex of the specified size made up of the
       * next scalars.  
       *
       * <p>See <code>stan::prob::simplex_validate(Eigen::Matrix)</code>.
       *
       * @param k Size of returned simplex.
       * @return Simplex read from the specified size number of scalars.
       */
      Matrix<T,Dynamic,1> simplex(unsigned int k) {
	Matrix<T,Dynamic,1> theta(vector(k));
	assert(stan::prob::simplex_validate(theta));
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
      Matrix<T,Dynamic,1> simplex_constrain(unsigned int k) {
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
       * @param lp Log probability to increment with log absolute Jacobian determinant.
       * @return The next simplex of the specified size.
       */
      Matrix<T,Dynamic,1> simplex_constrain(unsigned int k, T& lp) {
	return stan::prob::simplex_constrain(vector(k-1),lp);
      }

      /**
       * Return the next vector of specified size containing positive
       * values in order.  
       *
       * <p>See <code>stan::prob::pos_ordered_validate(T)</code>.
       *
       * @param k Size of returned vector.
       * @return Vector of positive values in ascending order.
       */
      Matrix<T,Dynamic,1> pos_ordered(unsigned int k) {
	Matrix<T,Dynamic,1> x(vector(k));
	assert(stan::prob::pos_ordered_validate(x));
	return x;
      }

      /**
       * Return the next positive, ordered vector of the specified
       * length.  
       *
       * <p>See <code>stan::prob::pos_ordered_constrain(Matrix)</code>.
       * 
       * @param k Length of returned vector.
       * @return Next positive, ordered vector of the specified
       * length.
       */
      Matrix<T,Dynamic,1> pos_ordered_constrain(unsigned int k) {
	return stan::prob::pos_ordered_constrain(vector(k));
      }

      /**
       * Return the next positive ordered vector of the specified size, incrementing
       * the specified reference with the log absolute Jacobian of the determinant.
       *
       * <p>See <code>stan::prob::pos_ordered_constrain(Matrix,T&)</code>.
       *
       * @param k Size of vector.
       * @param lp Log probability reference to increment.
       * @return Next positive ordered vector of the specified size.
       */
      Matrix<T,Dynamic,1> pos_ordered_constrain(unsigned int k, T& lp) {
	return stan::prob::pos_ordered_constrain(vector(k),lp);
      }


      /**
       * Returns the next correlation matrix of the specified dimensionality.
       *
       * <p>See <code>stan::prob::corr_matrix_validate(Matrix)</code>.
       *
       * @param k Dimensionality of correlation matrix.
       * @return Next correlation matrix of the specified dimensionality.
       */
      Matrix<T,Dynamic,Dynamic> corr_matrix(unsigned int k) {
	Matrix<T,Dynamic,Dynamic> x(matrix(k,k));
	assert(stan::prob::corr_matrix_validate(x));
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
      Matrix<T,Dynamic,Dynamic> corr_matrix_constrain(unsigned int k) {
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
      Matrix<T,Dynamic,Dynamic> corr_matrix_constrain(unsigned int k, T& lp) {
	return stan::prob::corr_matrix_constrain(vector((k * (k - 1)) / 2),k,lp);
      }


      /**
       * Return the next covariance matrix with the specified 
       * dimensionality.  
       *
       * <p>See <code>stan::prob::cov_matrix_validate(Matrix)</code>.
       *
       * @param k Dimensionality of covariance matrix.
       * @return Next covariance matrix of the specified dimensionality.
       */
      Matrix<T,Dynamic,Dynamic> cov_matrix(unsigned int k) {
	Matrix<T,Dynamic,Dynamic> y(matrix(k,k));
	assert(stan::prob::cov_matrix_validate(y));
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
      Matrix<T,Dynamic,Dynamic> cov_matrix_constrain(unsigned int k) {
	return stan::prob::cov_matrix_constrain(vector(k + (k * (k - 1)) / 2),k);
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
      Matrix<T,Dynamic,Dynamic> cov_matrix_constrain(unsigned int k, T& lp) {
	return stan::prob::cov_matrix_constrain(vector(k + (k * (k - 1)) / 2),k,lp);
      }



    };

  }

}

#endif
