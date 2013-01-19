#include <ostream>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

using std::vector;
using std::string;
using std::endl;
using std::pair;

vector<string> lookup_argument(const string& argument) {
  using boost::iequals;
  vector<string> args;
  if (iequals(argument, "int")) {
    args.push_back("int");
  } else if (iequals(argument, "ints")) {
    args.push_back("int");
    args.push_back("std::vector<int>");
    args.push_back("Eigen::Matrix<int, Eigen::Dynamic, 1>");
    args.push_back("Eigen::Matrix<int, 1, Eigen::Dynamic>");
  } else if (iequals(argument, "double")) {
    args.push_back("double");
    args.push_back("var");
  } else if (iequals(argument, "doubles")) {
    args.push_back("double");
    args.push_back("std::vector<double>");
    args.push_back("Eigen::Matrix<double, Eigen::Dynamic, 1>");
    args.push_back("Eigen::Matrix<double, 1, Eigen::Dynamic>");
    args.push_back("var");
    args.push_back("std::vector<var>");
    args.push_back("Eigen::Matrix<var, Eigen::Dynamic, 1>");
    args.push_back("Eigen::Matrix<var, 1, Eigen::Dynamic>");
  }
  return args;
}

std::ostream& operator<< (std::ostream& o, pair<string, string>& p) {
  o << "<" << p.first << ", " << p.second << ">" << endl;
  return o;
}

template <class T>
std::ostream& operator<< (std::ostream& o, vector<T>& vec) {
  o << "vector size: " << vec.size() << endl;
  for (size_t n = 0; n < vec.size(); n++) {
    o << "  \'" << vec[n] << "\'" << endl;
  }
  return o;
}



void write_includes(std::ostream& out, const string& include) {
  out << "#include <gtest/gtest.h>" << endl;
  out << "#include <boost/mpl/vector.hpp>" << endl;
  out << "#include <test/agrad/distributions/test_fixture.hpp>" << endl;
  out << "#include <" << include.substr(include.find("src/")+4) << ">" << endl;  
  out << endl;
}

vector<string> tokenize_arguments(const string& arguments) {
  vector<string> tokens;
  string delimiters = ", ";
  string args_only_string = arguments.substr(arguments.find(":") + 1);
  boost::algorithm::trim(args_only_string);
  boost::algorithm::split(tokens, args_only_string, 
			  boost::is_any_of(delimiters), 
			  boost::token_compress_on);
  return tokens;
}

size_t size(const vector<vector<string> >& sequences) {
  if (sequences.size() == 0)
    return 0;
  size_t N = 1;
  for (size_t n = 0; n < sequences.size(); n++) 
    N *= sequences[n].size();
  return N;
}

bool is_argument_list(const string& line) {
  size_t comment = line.find("//");
  if (comment == string::npos)
    return false;
  size_t keyword = line.find("Arguments:", comment+1);
  if (keyword == string::npos)
    return false;
  return true;
}

string read_arguments_from_file(const string& in_name) {
  string arguments;
  std::ifstream in(in_name.c_str());
  if (in.is_open() && in.good()) {
    std::getline (in, arguments);
    while (in.good() && !is_argument_list(arguments)) {
      std::getline (in, arguments);
    }
    in.close();
  }
  if (!is_argument_list(arguments))
    arguments = "";
  return arguments;
}

vector<pair<string, string> > read_test_names_from_file(const string& in_name) {
  vector<pair<string, string> > names;
  std::ifstream in(in_name.c_str());
  
  string file;
  in.seekg(0, std::ios::end);
  file.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&file[0], file.size());
  in.close();
  
  size_t pos = 0;
  string class_keyword = "class ";
  string public_keyword = "public ";
  while (pos < file.size()) {
    pos = file.find(class_keyword, pos);
    if (pos < file.size()) {
      pos += class_keyword.size();
      size_t pos2 = file.find(":", pos);
      string test_name = file.substr(pos, pos2-pos);
      pos = file.find(public_keyword, pos) + public_keyword.size();
      pos2 = file.find("{", pos);
      string fixture_name = file.substr(pos, pos2-pos);
      pos = file.find("};", pos) + 2;
      boost::algorithm::trim(test_name);
      boost::algorithm::trim(fixture_name);
      
      if (fixture_name.find("Test") != string::npos) {
	fixture_name += "Fixture";
	names.push_back(pair<string, string>(test_name, fixture_name));
      }
    }
  }
  return names;
}

vector<vector<string> > build_argument_sequence(const string& arguments) {
  vector<string> argument_list = tokenize_arguments(arguments);
  vector<vector<string> > argument_sequence;
  for (size_t n = 0; n < argument_list.size(); n++)
    argument_sequence.push_back(lookup_argument(argument_list[n]));
  return argument_sequence;
}

void write_types_typedef(std::ostream& out, string base, size_t& N, vector<vector<string> > argument_sequence, const size_t depth) {
  vector<string> args = argument_sequence.front();
  argument_sequence.erase(argument_sequence.begin());
  if (argument_sequence.size() > 0) {
    for (size_t n = 0; n < args.size(); n++)
      write_types_typedef(out, base + args[n] + ", ", N, argument_sequence, depth);
  } else {
    string extra_args;
    for (size_t n = depth; n < 10; n++) {
      extra_args += ", empty";
    }
    for (size_t n = 0; n < args.size(); n++) {
      out << "typedef boost::mpl::vector<" << base << args[n] << extra_args;
      if (extra_args.size() == 0)
	out << " ";
      out << "> type_" << N << ";" << endl;
      N++;
    }
  }
}

void write_types(std::ostream& out, const vector<vector<string> >& argument_sequence) {
  size_t N = 0;
  write_types_typedef(out, "", N, argument_sequence, argument_sequence.size());
  out << endl;
}

void write_test(std::ostream& out, const string& test_name, const string& fixture_name, const size_t N) {
  for (size_t n = 0; n < N; n++)
    out << "typedef boost::mpl::vector<" << test_name << ", type_" << n << "> " << test_name << "_" << n << ";" << endl;
  out << endl;
  for (size_t n = 0; n < N; n++)
    out << "INSTANTIATE_TYPED_TEST_CASE_P(" << test_name << "_" << n << ", " << fixture_name << ", " <<  test_name << "_" << n << ");" << endl;
  out << endl;
}

void write_test_cases(std::ostream& out, const string& in_name) {
  string arguments = read_arguments_from_file(in_name);
  vector<pair<string, string> > names = read_test_names_from_file(in_name);
  vector<vector<string> > argument_sequence = build_argument_sequence(arguments);
  
  write_types(out, argument_sequence); 
  for (size_t n = 0; n < names.size(); n++) {
    string test_name = names[n].first;
    string fixture_name = names[n].second;
    write_test(out, test_name, fixture_name, size(argument_sequence));
  }
}

/** 
 * Generate test cases.
 * 
 * @param argc Number of arguments
 * @param argv Arguments. Should contain one argument with a filename.
 * 
 * @return 0 for success, negative number otherwise.
 */
int main(int argc, const char* argv[]) {
  if (argc != 2)
    return -1;
  int return_code = 0;
  string in_suffix = "_test.hpp";
  string out_suffix = "_generated_test.cpp";

  string in_name = argv[1];
  
  size_t last_in_suffix = in_name.find_last_of(in_suffix) + 1 - in_suffix.length();
  string out_name = in_name.substr(0, last_in_suffix) + out_suffix;
  
  std::ofstream out(out_name.c_str());

  write_includes(out, in_name);
  write_test_cases(out, in_name);

  out.close();
  
  return 0;
}
