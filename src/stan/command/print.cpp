#include <iostream>
#include <iomanip>
#include <stan/mcmc/chains.hpp>

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
  if (argc == 1)
    return 0;


  const size_t skip = 3U;  // number of cols to skip up front.
  
  std::vector<std::string> names;
  std::vector<std::vector<size_t> > dimss;
  stan::mcmc::read_variables(argv[1], skip,
			     names, dimss);
  
  // test each file for correctness
  for (int i = 2; i <= argc; i++) {
    std::vector<std::string> curr_names;
    std::vector<std::vector<size_t> > curr_dimss;
    
    stan::mcmc::read_variables(argv[1], skip,
			       curr_names, curr_dimss);
    if (names.size() != curr_names.size()) {
      std::cerr << "names size doesn't match for file: " << argv[1] << " and " << argv[i] << std::endl;
      return -1;
    }
    for (size_t j = 0; j < names.size(); j++) {
      if (names[j] != curr_names[j]) {
	std::cerr << "names don't match for file: " << argv[1] << " and " << argv[i] << std::endl
		  << "  variable " << j << ": " << names[j] << " and " << curr_names[j] << std::endl;
	return -1;
      }
    }


    if (dimss.size() != curr_dimss.size()) {
      std::cerr << "dimss don't match for file: " << argv[1] << " and " << argv[i] << std::endl;
      return -1;
    }
    for (size_t j = 0; j < dimss.size(); j++) {
      if (dimss[j].size() != curr_dimss[j].size()) {
	std::cerr << "dimss size don't match for file: " << argv[1] << " and " << argv[i] << std::endl;
	return -1;
      }
      for (size_t k = 0; k < dimss[j].size(); k++) {
	if (dimss[j][k] != curr_dimss[j][k]) {
	  std::cerr << "dimss don't match for file: " << argv[1] << " and " << argv[i] << std::endl;
	  return -1;
	}
      }
    }
  }
  
  
  // read from each file and populate
  stan::mcmc::chains<> chains(argc-1, names, dimss);
  for (size_t n = 0; n < argc-1; n++) {
    stan::mcmc::add_chain(chains, n, argv[n+1], skip);
  }

  
  // print
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

  std::string model_name = "NEED MODEL NAME";
  size_t thin = 0;
  std::vector<size_t> name_lengths(names.size());

  for (size_t i = 0; i < names.size(); i++) {
    std::cout << "n: " << names[i] << std::endl;
    name_lengths[i] = names[i].length();
  }
  for (size_t i = 0; i < dimss.size(); i++) {
    if (dimss[i].size() == 1 && dimss[i][0] == 1) {
      name_lengths[i] += 0;
    } else {
      for (size_t j = 0; j < dimss[i].size(); j++) {
	name_lengths[i] += 2 + (dimss[i][j]+1) % 10;
      }
    }
  }

  size_t max_name_length = *std::max_element(name_lengths.begin(), 
					     name_lengths.end());
  
  std::cout << "Inference for Stan model: " << model_name << std::endl
	    << chains.num_chains() << " chains: each with iter=" << chains.num_samples()/chains.num_chains()
	    << "; warmup=" << chains.num_warmup_samples() << "; thin=" << thin 
	    << "; " << chains.num_samples() << " iterations saved." << std::endl;
  size_t idx = 0;

  for (size_t i = 0; i < dimss.size(); i++) {
    for (size_t j = 0; j < dimss[i].size(); j++) {
      std::cout << "dims[i][j]: " << dimss[i][j] << std::endl;
    }
    std::cout << std::endl;
  }
  /*for (size_t i = 0; i < dimss.size(); i++) {
    for (size_t j = 0; j < dimss[i].size(); j++) {
      
      for (size_t k = 0; k < dimss[i][j]; k++) {
	std::cout.width(max_name_length);
	std::cout.setf(std::ios::left);
	std::cout << names[i];
	if (dimss[i][j] > 1)
	  std::cout << "[" << j+1 << "]";
	std::cout << std::endl;
	
	idx++;
      }
    }
    }*/
  return 0;
}



