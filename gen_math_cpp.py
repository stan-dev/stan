import sys, re
import copy

types = {
    "int": ["double", "var"],
    "real": ["double", "var"],
    "vector": ["Eigen::VectorXd", "Eigen::Matrix<var, -1, 1>"],
    "matrix": ["Eigen::MatrixXd", "Eigen::Matrix<var, -1, -1>"]
}

def wrap_vec(t, times):
    if times == -1: return t
    return "std::vector<" + wrap_vec(t, times - 1) + ">"

for t in types.keys():
    for i in range(10):
        types[t + "[" + "," * i +"]"] = [wrap_vec(tt, i) for tt in types[t]]

return_types = copy.deepcopy(types)
for t, tt in return_types.items():
    return_types[t] = tt + [x.replace("int", "var") for x in tt]

for t in types:
    types[t] = ["const " + tt + "&" for tt in types[t]]

for t in types.keys():
    types[t + ","] = [tt + "," for tt in types[t]]

def merge_punctuation(split_def):
    i = 0
    while i < len(split_def):
        if split_def[i] in set(["(", ")", ","]):
            split_def[i-1] += split_def[i] + split_def[i+1]
            del split_def[i+1]
            del split_def[i]
        else:
            i+=1

seen_sigs = set()

def to_cpp(tokens):
    #unroll first token as it is the return type and we must just use the promoted type
    ret_type = return_types[tokens[0]]
    del tokens[0]

    mapped = [[]]
    for token in tokens:
        mapped = [m + [mt] for m in mapped
                  for mt in types.get(token, [token])]
    ret = ""
    for m in mapped:
        merge_punctuation(m)
        argstr =  " ".join(m) + ";\n"
        if argstr in seen_sigs:
            continue
        else:
            seen_sigs.add(argstr)
        ret_type_str = ""
        if "var" in argstr:
            ret_type_str = ret_type[1]
        else:
            ret_type_str = ret_type[0]
        ret += "template " + ret_type_str + " " + argstr
    return ret

def flatten1(l):
    return [x for e in l for x in e]

def split_and_retain(delimiter):
    def sr(s):
        if s in types:
            return [s]
        ret = []
        splits = s.split(delimiter)
        for i in range(len(splits) - 1):
            ret.append(splits[i])
            ret.append(delimiter)
        ret.append(splits[-1])
        return ret
    return sr

def to_tokens(line):
    return reduce(lambda l, f: flatten1(map(f, l)),
                  [split_and_retain("("),
                   split_and_retain(")"),
                   split_and_retain(",")],
                  line.split())

stdlib_fns = [
    "is_nan(real)", "abs(int)", "abs(real)", "add(int, int)", "add(real, real)",
    "add(int)", "add(real)", "atan2(real, real)"
]
stdlib_fn_re_pattern = "|".join("(.*" + re.escape(s) + ".*)" for s in stdlib_fns)
stdlib_fn_re = re.compile(stdlib_fn_re_pattern)
def is_stdlib_fn(line):
    return re.search(stdlib_fn_re, line)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        #t = to_tokens("real sars(int[,,,,,,,])")
        #print sorted(types.keys())
        #print('tokens:', t)
        #print("cpp: ", to_cpp(t))
        sys.exit(0)
    print("#include <stan/math/rev/mat.hpp>")
    print("namespace stan {")
    print("namespace math {")
    for line in sys.stdin:
        if "row vector" in line:
            continue
        if "_rng" in line: # XXX
            continue
        if is_stdlib_fn(line):
            continue
        print(to_cpp(to_tokens(line)))
    print("} // namespace math")
    print("} // namespace stan")
