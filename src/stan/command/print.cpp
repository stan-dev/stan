#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ios>
#include <stan/mcmc/chains.hpp>

int calculate_size(const Eigen::VectorXd& x, const std::string& name) {
  using std::max;
  using std::ceil;
  using std::log10;

  double size = ceil(log10(x.maxCoeff()+0.001)) + 2.0;
  if (x.minCoeff() < 0)
    size = max(size, ceil(log10(-x.minCoeff()+0.01))+3.0);

  return max(size,
	     max(name.length(), std::string("-0.0").length())+0.0);
}

/** 
 * The Stan print function.
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * 
 * @return 0 for success, 
 *         non-zero otherwise
 */
int main(int argc, const char* argv[]) {
  if (argc == 1) {
    std::cout << "usage: print <filename 1> <filename 2>";
    return 0;
  }

  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i++) {
    filenames.push_back(argv[i]);
  }
  
  Eigen::VectorXi thin(filenames.size());
  
  std::ifstream ifstream;
  ifstream.open(filenames[0].c_str());
  stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
  stan::mcmc::chains<> chains(stan_csv);
  ifstream.close();
  thin(0) = stan_csv.metadata.thin;
  

  for (int chain = 1; chain < filenames.size(); chain++) {
    ifstream.open(filenames[chain].c_str());
    stan_csv = stan::io::stan_csv_reader::parse(ifstream);
    chains.add(stan_csv);
    ifstream.close();
    thin(chain) = stan_csv.metadata.thin;
  }

  // print  
  const int skip = 3;
  std::string model_name = "NEED MODEL NAME";
  int max_name_length = 0;
  for (int i = skip; i < chains.num_params(); i++) 
    if (chains.param_name(i).length() > max_name_length)
      max_name_length = chains.param_name(i).length();

  Eigen::MatrixXd values(chains.num_params(),10);
  values.setZero();
  Eigen::VectorXd probs(5);
  probs << 0.025, 0.25, 0.5, 0.75, 0.975;

  for (int i = skip; i < chains.num_params(); i++) {
    values(i,0) = chains.mean(i);
    values(i,1) = chains.sd(i) / std::sqrt(chains.num_kept_samples());
    values(i,2) = chains.sd(i);
    Eigen::VectorXd quantiles = chains.quantiles(i,probs);
    for (int j = 0; j < 5; j++)
      values(i,3+j) = quantiles(j);
    values(i,8) = chains.effective_sample_size(i);
    values(i,9) = chains.split_potential_scale_reduction(i);
  }
  
  Eigen::VectorXi column_lengths(11);
  column_lengths(0) = max_name_length + 1;
  column_lengths(1) = calculate_size(values.col(0), "mean")+1;
  column_lengths(2) = calculate_size(values.col(1), "se_mean")+1;
  column_lengths(3) = calculate_size(values.col(2), "sd")+1;
  column_lengths(4) = calculate_size(values.col(3), "2.5%")+1;
  column_lengths(5) = calculate_size(values.col(4), "25%")+1;
  column_lengths(6) = calculate_size(values.col(5), "50%")+1;
  column_lengths(7) = calculate_size(values.col(6), "75%")+1;
  column_lengths(8) = calculate_size(values.col(7), "97.5%")+1;
  column_lengths(9) = calculate_size(values.col(8), "n_eff")+1;
  column_lengths(10) = calculate_size(values.col(9), "Rhat")+1;
  
  std::cout << "Inference for Stan model: " << model_name << std::endl
	    << chains.num_chains() << " chains: each with iter=(" << chains.num_kept_samples(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << chains.num_kept_samples(chain);
  std::cout << ")";
  std::cout << "; warmup=(" << chains.warmup(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << chains.warmup(chain);
  std::cout << ")";
  std::cout << "; thin=(" << thin(0);
  for (int chain = 1; chain < chains.num_chains(); chain++)
    std::cout << "," << thin(chain);
  std::cout << ")";
  std::cout << "; " << chains.num_samples() << " iterations saved." 
	    << std::endl << std::endl;
  
  // header
  std::cout << std::setw(column_lengths(0)) << ""
	    << std::setw(column_lengths(1)) << "mean"
	    << std::setw(column_lengths(2)) << "se_mean" 
	    << std::setw(column_lengths(3)) << "sd" 
	    << std::setw(column_lengths(4)) << "2.5%" 
	    << std::setw(column_lengths(5)) << "25%" 
	    << std::setw(column_lengths(6)) << "50%" 
	    << std::setw(column_lengths(7)) << "75%" 
	    << std::setw(column_lengths(8)) << "97.5%" 
	    << std::setw(column_lengths(9)) << "n_eff" 
	    << std::setw(column_lengths(10)) << "Rhat" 
	    << std::endl;
  // each row
  for (int i = skip; i < chains.num_params(); i++) {
    std::cout << std::setw(column_lengths(0)) << std::left << chains.param_name(i)
	      << std::right << std::fixed << std::setprecision(1)
	      << std::setw(column_lengths(1)) << chains.mean(i)
	      << std::setw(column_lengths(2)) << chains.sd(i) / std::sqrt(chains.num_kept_samples())
	      << std::setw(column_lengths(3)) << chains.sd(i)
	      << std::setw(column_lengths(4)) << chains.quantile(i,0.025)
	      << std::setw(column_lengths(5)) << chains.quantile(i,0.25)
	      << std::setw(column_lengths(6)) << chains.quantile(i,0.5)
	      << std::setw(column_lengths(7)) << chains.quantile(i,0.75)
	      << std::setw(column_lengths(8)) << chains.quantile(i,0.975)
	      << std::setw(column_lengths(9)) << chains.effective_sample_size(i)
	      << std::setw(column_lengths(10)) << chains.split_potential_scale_reduction(i)
	      << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Samples were drawn using " << stan_csv.adaptation.sampler << "." << std::endl
	    << "For each parameter, n_eff is a crude measure of effective sample size," << std::endl
	    << "and Rhat is the potential scale reduction factor on split chains (at " << std::endl
	    << "convergence, Rhat=1)." << std::endl
	    << std::endl;

  /*
Inference for Stan model: schools_code.
1 chains: each with iter=100; warmup=50; thin=1; 100 iterations saved.

         mean se_mean   sd  2.5%  25%  50%  75% 97.5% n_eff Rhat
mu        8.3     0.6  4.0   1.6  5.2  8.1 11.0  15.8    42  1.0
tau       6.2     1.0  5.1   0.1  2.5  5.3  8.6  17.9    28  1.0
eta[1]    0.5     0.2  1.1  -1.8 -0.1  0.5  1.3   2.4    50  1.0
eta[2]    0.0     0.1  0.8  -1.5 -0.5  0.0  0.6   1.4    50  1.0
eta[3]   -0.3     0.1  0.7  -1.6 -0.8 -0.3  0.2   1.1    26  1.1
eta[4]   -0.1     0.1  0.7  -1.6 -0.5 -0.1  0.2   1.5    50  1.0
eta[5]   -0.4     0.1  0.8  -1.6 -1.1 -0.4  0.1   1.2    49  1.0
eta[6]   -0.2     0.1  0.9  -1.9 -1.0 -0.5  0.5   1.6    50  1.0
eta[7]    0.3     0.2  0.9  -1.8 -0.2  0.6  0.9   1.7    27  1.0
eta[8]    0.0     0.1  0.7  -1.4 -0.4  0.1  0.6   1.1    50  1.0
theta[1] 12.9     1.5 10.4  -6.5  7.4 11.4 15.1  39.1    50  1.0
theta[2]  9.2     1.0  6.1   0.0  4.3  8.4 13.3  22.2    35  1.0
theta[3]  6.3     0.9  6.4 -12.2  3.9  6.7 10.2  13.3    50  1.0
theta[4]  8.0     1.0  6.8  -6.9  4.6  7.8 12.6  19.4    50  1.0
theta[5]  4.6     0.8  5.6  -6.4  1.2  6.0  8.9  13.0    50  1.0
theta[6]  6.2     1.0  5.2  -1.9  2.1  6.8 10.5  13.6    27  1.0
theta[7] 11.0     0.8  5.1   2.0  7.9 10.8 14.3  21.0    40  1.0
theta[8]  8.4     1.0  6.2  -3.3  5.3  8.4 13.0  17.8    41  1.0
lp__     -4.5     0.8  2.7 -11.0 -5.7 -4.3 -2.4  -0.7    13  1.1

Samples were drawn using NUTS2 at Sat Feb 16 00:42:14 2013.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
   */
  

  return 0;
}



