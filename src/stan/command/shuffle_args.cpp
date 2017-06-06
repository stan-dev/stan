#include <algorithm>
#include <ctime>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>


/**
 * Mersenne Twister RNG (global).
 */
boost::random::mt19937 gen_;

/**
 * Return non-negative random number strictly less than the upper
 * bound value.
 *
 * @param upper_bound Upper bound (exclusive) for return value.
 * @return Random non-negative value less than i.
 */
int range_rng(int upper_bound) {
  boost::random::uniform_int_distribution<> dist(0, upper_bound - 1);
  return dist(gen_);
}

/**
 * Print a shuffled version of the arguments (excluding executable
 * name) to <code>std::cout</code>.  
 *
 * <p>Uses <code>std::time(0)</code> to seed the
 * global RNG <code>gen_</code>.
 * 
 * @param argc Number of arguments (including executable name)
 * @param argv Arguments
 * @return 0 if successful, -1 if not.
 */
int main(int argc, char* argv[]) {
  gen_.seed(std::time(0));
  std::random_shuffle(argv + 1, argv + argc, range_rng); 
  for (int i = 1; i < argc; ++i) {
    if (i > 0) 
      std::cout << " ";
    std::cout << argv[i];
  }
  std::cout << std::endl;
  return 0;
}
