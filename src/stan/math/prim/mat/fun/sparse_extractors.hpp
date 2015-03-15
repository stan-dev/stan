#ifndef STAN__MATH__MATRIX_SPARSE_EXTRACTORS_HPP
#define STAN__MATH__MATRIX_SPARSE_EXTRACTORS_HPP

#include <Eigen/Sparse>
#include <vector>
#include <numeric>

namespace stan {

	namespace math {
    
    template <typename _Scalar> 
    const Eigen::Matrix<_Scalar, Eigen::Dynamic,1> extract_w(Eigen::SparseMatrix<_Scalar> A) {
    	Eigen::Matrix<_Scalar,Eigen::Dynamic,1> w(A.nonZeros());
    	for ( int j = 0; j < A.nonZeros(); ++j)
    		w[j] = *(A.valuePtr()+j);
    	return w;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_v(Eigen::SparseMatrix<_Scalar> A) {
    	std::vector<int> v(A.nonZeros());
    	for ( int j = 0; j < A.nonZeros(); ++j)
    		v[j] = *(A.innerIndexPtr()+j) + 1;   // make 1-indexed
    	return v;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_u(Eigen::SparseMatrix<_Scalar> A) {
    	std::vector<int> u(A.outerSize()+1);
    	for ( int j = 0; j <= A.outerSize(); ++j)
    		u[j] = *(A.outerIndexPtr()+j) + 1; // make 1-indexed
    	return u;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_z(Eigen::SparseMatrix<_Scalar> A) {
    	std::vector<int> u(A.outerSize()+1);
    	std::vector<int> z(A.outerSize()+1);
    	u = extract_u(A);
    	std::adjacent_difference(u.begin(), u.end(), z.begin());
    	z.erase(z.begin());
    	return z;
    }
    
    
    template <typename _Scalar> 
    const Eigen::Matrix<_Scalar,Eigen::Dynamic,1> extract_w(Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> A) {
    	Eigen::Matrix<_Scalar,Eigen::Dynamic,1> w(A.nonZeros());
    	for ( int j = 0; j < A.nonZeros(); ++j)
    		w[j] = *(A.valuePtr()+j);
    	return w;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_v(Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> A) {
    	std::vector<int> v(A.nonZeros());
    	for ( int j = 0; j < A.nonZeros(); ++j)
    		v[j] = *(A.innerIndexPtr()+j) + 1;  // make 1-indexed
    	return v;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_u(Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> A) {
    	std::vector<int> u(A.outerSize()+1);
    	for ( int j = 0; j <= A.outerSize(); ++j)
    		u[j] = *(A.outerIndexPtr()+j) + 1;  // make 1-indexed
    	return u;
    }
    
    template <typename _Scalar> 
    const std::vector<int> extract_z(Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> A) {
    	std::vector<int> u(A.outerSize()+1);
    	std::vector<int> z(A.outerSize()+1);
    	u = extract_u(A);
    	std::adjacent_difference(u.begin(), u.end(), z.begin());
    	z.erase(z.begin());
    	return z;
    }
    
	}
}

#endif
