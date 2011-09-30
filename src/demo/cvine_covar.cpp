#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

/* This function to parameterize a correlation matrix is based on
 * 
 * Daniel Lewandowski, Dorota Kurowicka, and Harry Joe, 
 * Generating random correlation matrices based on vines and extended onion method
 * Journal of Multivariate Analysis 100 (2009) 1989–2001
 * 
 * In particular, their C-vine method using a new result based on the Cholesky factorization
 *
 * @author Ben Goodrich
*/

// NOTE: Be sure to keep in mind whether the parameters are in the unbounded or transformed state

// not called directly unless we want to put priors on cpcs rather than matrix
template <typename T>
Matrix<T,Dynamic,Dynamic>
read_corr_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
		 const unsigned int& K) {
    
  // Note: Would be faster to preallocate these once and pass them into this function
  Array<T,Dynamic,1> temp;         // temporary holder
  Array<T,Dynamic,1> acc(K-1);     // accumlator of products
  acc.setOnes();
  Array<T,Dynamic,Dynamic> L(K,K); // Cholesky factor of correlation matrix
  L.setZero();

  unsigned int position = 0;
  unsigned int pull = K - 1;

  L(0,0) = 1.0;
  L.col(0).tail(pull) = temp = CPCs.head(pull);
  acc.tail(pull) = 1.0 - temp.square();
  for(unsigned int i = 1; i < (K - 1); i++) {
    position += pull;
    pull--;
    temp = CPCs.segment(position, pull);
    L(i,i) = sqrt(acc(i-1));
    L.col(i).tail(pull) = temp * acc.tail(pull).sqrt();
    acc.tail(pull) *= 1.0 - temp.square();
  }
  L(K-1,K-1) = sqrt(acc(K-2));
  return L.matrix().template triangularView<Eigen::Lower>() * L.matrix().transpose();
  // Note: There is room to optimize this ^^^ special matrix product further; see dsyrk, dtrmm, and dlauum in BLAS
}

// Takes into account the Jacobian determinant of the mapping: CPCs -> correlations
template <typename T>
Matrix<T,Dynamic,Dynamic>
read_corr_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
		 const unsigned int& K,
		 T& log_prob) {

  unsigned int k = 0; 
  unsigned int i = 0;
  unsigned int counter = 0;
  T log_1cpc2;
  T lead = K - 2.0;
  // NOTE: no need to abs() because this Jacobian determinant is strictly positive (and triangular)
  for(unsigned int j = 0; j < (CPCs.rows() - 1); j++) {
    log_1cpc2 = log(1.0 - pow(CPCs[counter], 2));
    log_prob += lead / 2.0 * log_1cpc2; // derivative of correlation wrt CPC
    i++;
    if(i > K) {
      k++;
      i = k + 1;
      lead = K - k - 1.0;
    }
    counter++;
  }
  return read_corr_matrix(CPCs, K);
}

// Builds a covariance matrix from CPCs and standard deviations
template<typename T>
Matrix<T,Dynamic,Dynamic>
read_cov_matrix(const Array<T,Dynamic,1>& CPCs,    // on (-1,1)
                const Array<T,Dynamic,1>& sds) {   // on (0,inf)

  unsigned int K = sds.rows();
  DiagonalMatrix<T,Dynamic> D(K);
  D.diagonal() = sds;
  return D * read_corr_matrix(CPCs, K) * D;
}

// this is the main one we'll call from outside for covariance transforms
// ---------------------------
template <typename T>
Matrix<T,Dynamic,Dynamic>
read_cov_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
		const Array<T,Dynamic,1>& sds,  // on (0,inf)
		T& log_prob) {
  unsigned int K = sds.rows();
  unsigned int counter = 0;
  const Array<T,Dynamic,1> log_sds = sds.log();
  // (diagonal and positive) Jacobian determinant for the mapping: correlations -> covariances
  for(unsigned int i = 0; i < (K - 1); i++) for(unsigned int j = i + 1; j < K; j++) {
    log_prob += log_sds(i,1) + log_sds(j,1);
  }
  DiagonalMatrix<T,Dynamic> D(K);
  D.diagonal() = sds;
  return D * read_corr_matrix(CPCs, K, log_prob) * D;
}

// This function calculates the degrees of freedom for the t distribution
// that corresponds to the shape parameter in the Lewandowski et. al. distribution
template<typename T>
const Array<T,Dynamic,1>
make_nu(const T eta,             // hyperparameter on (0,inf), eta = 1 <-> correlation matrix is uniform
	const unsigned int K) {  // number of variables in covariance matrix
  
  Array<T,Dynamic,1> nu(K * (K - 1) / 2);
  
  T alpha = eta + (K - 2.0) / 2.0; // from Lewandowski et. al.
  // Best (1978) implies nu = 2 * alpha for the dof in a t distribution that generates a beta variate on (-1,1)
  T alpha2 = 2.0 * alpha; 
  
  for(unsigned int j = 0; j < (K - 1); j++) {
    nu(j) = alpha2;
  }
  
  unsigned int counter = K - 1;
  for(unsigned int i = 1; i < (K - 1); i++) {
    alpha -= 0.5;
    alpha2 = 2.0 * alpha;
    for(unsigned int j = i + 1; j < K; j++) {
      nu(counter) = alpha2;
      counter++;
    }
  }
  
  return nu;
}

// This function is intended to make starting values, given a covariance matrix Sigma
// The transformations are hard coded as log for standard deviations and Fisher
// transformations of CPCs
template<typename T>
bool
factor_cov_matrix(Array<T,Dynamic,1>& CPCs, // will fill this unbounded
		  Array<T,Dynamic,1>& sds,  // will fill this unbounded
		  const MatrixXd& Sigma) {

  unsigned int K = sds.rows();

  sds = Sigma.diagonal().array();
  if( (sds <= 0).any() ) return false;
  sds = sds.sqrt();
  
  DiagonalMatrix<double,Dynamic> D(K);
  D.diagonal() = sds.inverse();
  sds = sds.log(); // now unbounded
  
  MatrixXd R = D * Sigma * D;
  R.diagonal().setOnes(); // to hopefully prevent pivoting due to floating point error
  LDLT<MatrixXd> ldlt;
  ldlt = R.ldlt();
  if( !ldlt.isPositive() ) return false;
  MatrixXd U = ldlt.matrixU();

  unsigned int position = 0;
  unsigned int pull = K - 1;

  Array<T,Dynamic,1> temp = U.row(0).tail(pull);
  CPCs.head(pull) = temp;
  
  Array<T,Dynamic,1> acc(K);
  acc(0) = -0.0;
  acc.tail(pull) = 1.0 - temp.square();
  for(unsigned int i = 1; i < (K - 1); i++) {
    position += pull;
    pull--;
    temp = U.row(i).tail(pull).array();
    temp /= sqrt(acc.tail(pull) / acc(i));
    CPCs.segment(position, pull) = temp;
    acc.tail(pull) *= 1.0 - temp.square();
  }
  CPCs = 0.5 * ( (1.0 + CPCs) / (1.0 - CPCs) ).log(); // now unbounded
  return true;
}

int main () {   
  srand ( time(NULL) );
  const unsigned int K = 5;
  const double eta = 1.1;
  const ArrayXd nu = make_nu(eta, K);
  std::cout << "nu is:" << std::endl << nu.matrix().transpose() << std::endl;
  ArrayXd CPCs(K * (K - 1) / 2);
  CPCs.setRandom(); // bounded
  CPCs = 0.5 * ( (1.0 + CPCs) / (1.0 - CPCs) ).log(); // unbounded
  
  ArrayXd sds(K);
  sds.setRandom(); // unbounded

  MatrixXd Sigma(K,K);
  std::cout << "For the mapping from unbounded parameters to a covariance matrix ..." << std::endl;
  std::cout << "unbounded CPCs originally is: " << std::endl << CPCs.matrix().transpose() << std::endl;
  std::cout << "unbounded sds originally is: " << std::endl << sds.matrix().transpose() << std::endl;
  
  CPCs = ((2 * CPCs).exp() - 1) / ((2 * CPCs).exp() + 1); // bounded
  sds = sds.exp(); // bounded
  
  Sigma = read_cov_matrix(CPCs, sds);
  std::cout << "Covariance matrix is: " << std::endl << Sigma << std::endl;

  std::cout << "For the roundtrip mapping from a covariance matrix back to unbounded parameters ..." << std::endl;
  bool valid = factor_cov_matrix(CPCs, sds, Sigma);
  if(valid) {
    std::cout << "Covariance matrix is positive definite" << std::endl;
    std::cout << "unbounded CPCs is now: " << std::endl << CPCs.matrix().transpose() << std::endl;
    std::cout << "unbounded sds is now: "  << std::endl << sds.matrix().transpose() << std::endl;
  }
  else {
    std::cout << "Covariance matrix is indefinite (indicating a bug)" << std::endl;
    return 1;
  }
  return 0;
}
