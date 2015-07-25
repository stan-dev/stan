#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <stan/lang/ast_def.cpp>
#include <vector>

#include <gtest/gtest.h>

TEST(lang, writeSignatures) {
  using std::fstream;
  using std::string;
  using std::set;
  using std::vector;
  using stan::lang::expr_type;
  using stan::lang::function_signatures;
  using stan::lang::function_signature_t;
  std::cout << "FUNCTION SIGNATURES: function_sigs.csv" << std::endl;
  std::cout << "SAMPLING SIGNATURES: sampling_sigs.csv" << std::endl;

  fstream fs("function_sigs.csv", std::fstream::out);
  fstream ss("sampling_sigs.csv", std::fstream::out);

  set<string> keys = function_signatures::instance().key_set();
  std::cout << "keys size=" << keys.size() << std::endl;
  for (set<string>::iterator it = keys.begin(); it != keys.end(); ++it) {
    string key = *it;
    const vector<function_signature_t> sigs_for_key
      = function_signatures::instance().sigs(key);
    std::cout << "key=" << key << " size=" << sigs_for_key.size() << std::endl;
    for (size_t i = 0; i < sigs_for_key.size(); ++i) {
      function_signature_t sig = sigs_for_key[i];
      vector<expr_type> arg_types = sig.second;
      fs << key;
      for (size_t j = 0; j < arg_types.size(); ++j)
        fs << ',' << arg_types[j];
      // expr_type result_type = sig.first;
      // fs << ',' << result_type;
      fs << std::endl;

      if (key.size() > 4 && key.substr(key.size() - 4) == "_log") {
        if (key == "multiply_log") continue;
        if (key == "binomial_coefficient_log") continue;
        if (key.size() > 7 && key.substr(key.size() - 7) == "cdf_log")
          continue;
        ss << key;
        for (size_t j = 0; j < arg_types.size(); ++j)
          ss << ',' << arg_types[j];
        ss << std::endl;
      }
    }
  }

    ss.close();
  fs.close();
  EXPECT_EQ(2, 1 + 1);
}
