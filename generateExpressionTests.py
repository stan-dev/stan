import re
import os
import sys
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter

if os.name == "nt":  # Windows
    make = "mingw32-make"
else:
    make = "make"

src_folder = "./src/test/unit/expressions/"
build_folder = "./test/unit/expressions/"
exceptions_list_location = "./lib/stan_math/test/expressions/stan_math_sigs_exceptions.expected"

args2test = ["matrix", "vector", "row_vector"]
arg_types = {
    'int': "int",
    'int[]': "std::vector<int>",
    'int[,]': "std::vector<std::vector<int>>",
    'real': "SCALAR",
    'real[]': "std::vector<SCALAR>",
    'real[,]': "std::vector<std::vector<SCALAR>>",
    'vector': "Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>",
    'vector[]': "std::vector<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>>",
    'row_vector': "Eigen::Matrix<SCALAR, 1, Eigen::Dynamic>",
    'row_vector[]': "std::vector<Eigen::Matrix<SCALAR, 1, Eigen::Dynamic>>",
    'matrix': "Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>",
    '(vector, vector, data real[], data int[]) => vector': "auto",
    "rng": "std::minstd_rand"
}


def get_ignored_signatures():
    part_sig = ""
    ignored = set()
    for signature in open(exceptions_list_location):
        if not signature.endswith(")\n"):
            part_sig += signature
            continue
        part_sig = ""
        ignored.add(signature)
    return ignored


def get_signatures():
    if os.name == "nt":
        stanc3 = "bin/stanc.exe"
    else:
        stanc3 = "bin/stanc"
    p = subprocess.Popen((make, stanc3))
    if p.wait() != 0:
        sys.stderr.write("Error in making stanc3!")
        sys.exit(-1)

    p = subprocess.Popen((stanc3 + " --dump-stan-math-signatures"), stdout=subprocess.PIPE, universal_newlines=True)

    res = []
    part_sig = ""
    for signature in p.stdout:
        if not signature.endswith(")\n"):
            part_sig += signature
            continue
        res.append(part_sig + signature)
        part_sig = ""

    if p.wait() != 0:
        sys.stderr.write("Error in getting signatures from stanc3!\n")
        sys.exit(-1)

    return res


def parse_signature(signature):
    return_type, rest = signature.split(" ", 1)
    function_name, rest = rest.split("(", 1)
    args = re.findall(r"(?:[(][^()]+[)][^,()]+)|(?:[^,()]+(?:,*[]])?)", rest)
    args = [i.strip() for i in args if i.strip()]
    return return_type, function_name, args


def make_arg_code(arg, scalar, var_name, function_name):
    arg_type = arg_types[arg].replace("SCALAR", scalar)
    if arg == '(vector, vector, data real[], data int[]) => vector':
        return "  %s %s = [](const auto& a, const auto&, const auto&, const auto&){return a;};" % (arg_type, var_name)
    elif function_name == "acosh":
        return "  %s %s = stan::math::as_array_or_scalar(stan::test::make_arg<%s>())+1;\n" % (arg_type, var_name, arg_type)
    elif function_name == "log1m_exp":
        return "  %s %s = stan::math::as_array_or_scalar(stan::test::make_arg<%s>())-1;\n" % (arg_type, var_name, arg_type)
    else:
        return "  %s %s = stan::test::make_arg<%s>();\n" % (arg_type, var_name, arg_type)


def save_tests_in_files(N_files, tests):
    for i in range(N_files):
        start = i * len(tests) // N_files
        end = (i + 1) * len(tests) // N_files
        with open(src_folder + "tests%d_test.cpp" % i, "w") as out:
            out.write("#include <test/unit/expressions/expression_test_helpers.hpp>\n\n")
            for test in tests[start:end]:
                out.write(test)


def main(functions=(), j=1):
    ignored = get_ignored_signatures()

    test_n = {}
    tests = []
    for signature in get_signatures():
        return_type, function_name, function_args = parse_signature(signature)
        if signature in ignored:
            continue
        for arg2test in args2test:
            if arg2test in function_args:
                break
        else:
            continue
        if functions and function_name not in functions:
            continue
        func_test_n = test_n.get(function_name, 0)
        test_n[function_name] = func_test_n + 1

        if function_name.endswith("_rng"):
            function_args.append("rng")

        test_code = ""
        for overload, scalar in (("Prim", "double"), ("Rev", "stan::math::var"), ("Fwd", "stan::math::fvar<double>")):
            if (function_name.endswith("_rng") and overload != "Prim"):
                continue
            test_code += "TEST(ExpressionTest%s, %s%d){\n" % (overload, function_name, func_test_n)

            for n, arg in enumerate(function_args):
                test_code += make_arg_code(arg, scalar, "arg_mat%d" % n, function_name)
            test_code += "  auto res_mat = stan::math::%s(" % function_name
            for n in range(len(function_args) - 1):
                test_code += "arg_mat%d, " % n
            test_code += "arg_mat%d);\n\n" % (len(function_args) - 1)

            for n, arg in enumerate(function_args):
                test_code += make_arg_code(arg, scalar, "arg_expr%d" % n, function_name)
                if arg in arg2test:
                    test_code += "  int counter%d = 0;\n" % n
                    test_code += "  stan::test::counterOp<%s> counter_op%d(&counter%d);\n" % (scalar, n, n)
            test_code += "  auto res_expr = stan::math::%s(" % function_name
            for n, arg in enumerate(function_args[:-1]):
                if arg in arg2test:
                    test_code += "arg_expr%d.unaryExpr(counter_op%d), " % (n, n)
                else:
                    test_code += "arg_expr%d, " % n
            if function_args[-1] in arg2test:
                test_code += "arg_expr%d.unaryExpr(counter_op%d));\n\n" % (len(function_args) - 1, len(function_args) - 1)
            else:
                test_code += "arg_expr%d);\n\n" % (len(function_args) - 1)

            test_code += "  EXPECT_STAN_EQ(res_expr, res_mat);\n"

            for n, arg in enumerate(function_args):
                if arg in arg2test:
                    if function_name == "rank":
                        test_code += '  EXPECT_LE(counter%d, 2);\n' % (n)
                    else:
                        test_code += '  EXPECT_LE(counter%d, 1);\n' % (n)
            if overload == "Rev" and (return_type.startswith("real") or
                                      return_type.startswith("vector") or
                                      return_type.startswith("row_vector") or
                                      return_type.startswith("matrix")):
                test_code += "  (stan::test::recursive_sum(res_mat) + stan::test::recursive_sum(res_expr)).grad();\n"
                for n, arg in enumerate(function_args):
                    if arg == '(vector, vector, data real[], data int[]) => vector':
                        continue
                    test_code += "  EXPECT_STAN_ADJ_EQ(arg_expr%d,arg_mat%d);\n" % (n, n)
            test_code += "}\n\n"
        tests.append(test_code)

    save_tests_in_files(j, tests)


def processCLIArgs():
    """
    Define and process the command line interface to the runTests.py script.
    """
    parser = ArgumentParser(description="Generate and run stan math expression tests.",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("-j", metavar="N", type=int, default=1,
                        help="Number of cores for make to use. Also number of files tests are split in.")
    parser.add_argument("functions", nargs="+", type=str, default=[],
                        help="Names of the functions to test. By default tests all functions.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = processCLIArgs()
    main(cli_args.functions, cli_args.j)
