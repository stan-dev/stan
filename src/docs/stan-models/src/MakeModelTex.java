
import java.io.*;
import java.util.*;
import java.util.regex.*;


// to run:
// pdflatex models-out.tex
// pdflatex models-out.tex
// makeidx models-out
// pdflatex models-out.tex

class MakeModelTex {
        
    static final char EOL = '\n';

    public static void main(String[] args) throws IOException {
        Writer texWriter 
            = new OutputStreamWriter(new FileOutputStream(args[0]),
                                     "ASCII");
        writeStart(texWriter);
        for (int i = 1; i < args.length; ++i)
            writeModels(new File(args[i]),texWriter);
        writeEnd(texWriter);
        texWriter.close();
    }

    static String toTexable(File f) {
        String s = f.getName().replace('_',' ');
        if (s.endsWith(".stan"))
            s = s.substring(0,s.length()-5);
        if (s.length() == 0) return "EMPTY NAME";
        StringBuilder sb = new StringBuilder();
        sb.append(Character.toTitleCase(s.charAt(0)));
        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) == ' ') {
                sb.append(' ');
                ++i;
                if (i < s.length())
                    sb.append(Character.toTitleCase(s.charAt(i)));
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    static void writeStart(Writer texWriter)
        throws IOException {

        texWriter.write("\\documentclass[11pt,openany]{book}" + EOL);
        texWriter.write("\\usepackage{models}" + EOL);
        texWriter.write("\\pagestyle{empty}" + EOL);
        texWriter.write("\\begin{document}" + EOL);
        texWriter.write("\\begin{center}" + EOL);
        texWriter.write("\\mbox{ }" + EOL);
        texWriter.write("\\\\[2in]" + EOL);
        texWriter.write("{\\Huge\\bf Stan Example Models}" + EOL);
        texWriter.write("\\\\[18pt]" + EOL);
        texWriter.write("{\\large Version 2.5.0}" + EOL);
        texWriter.write("\\\\[12pt]" + EOL);
        texWriter.write("{\\large \\today}" + EOL);
        texWriter.write("\\\\[1in]" + EOL);
        texWriter.write("{\\Large Stan Development Team}" + EOL);
        texWriter.write("\\vfill" + EOL);
        texWriter.write("\\end{center}" + EOL);

        texWriter.write("\\tableofcontents" + EOL);
        texWriter.write("\\pagestyle{plain}" + EOL);
    }

    static void writeModels(File partPath, Writer texWriter) 
        throws IOException {
        
        texWriter.write("\\part{" + toTexable(partPath) + "}" 
                        + EOL + EOL);

        for (File modelDir : partPath.listFiles()) {
            if (modelDir.getName().startsWith(".")) continue;
            texWriter.write("\\chapter{" + toTexable(modelDir) + "}"
                            + EOL + EOL);

            for (File modelFile : modelDir.listFiles()) {
                if (!modelFile.getName().endsWith(".stan")) continue;
                texWriter.write("\\section{" + toTexable(modelFile) + "}"
                                + EOL + EOL);
                texWriter.write("\\verbatiminput{"
                                + modelFile.getCanonicalPath()
                                + "}"
                                + EOL + EOL);
                addFunctionLabels(modelFile,texWriter);
            }
        }
    }

    static void addFunctionLabels(File f, Writer texWriter)
        throws IOException {
        BufferedReader buf 
            = new BufferedReader(new InputStreamReader(new FileInputStream(f),
                                                       "ASCII"));
        String line;
        while ((line = buf.readLine()) != null)
            addFunctionLabelsLine(line,texWriter);
        buf.close();
    }

    static String FUNCTION_REGEX = "([a-zA-Z0-9_]+)\\(";
    static Pattern FUNCTION_PATTERN = Pattern.compile(FUNCTION_REGEX);

    static String stripComments(String line) {
        int i = java.lang.Math.max(line.indexOf("#"),
                                   line.indexOf("//"));
        return i >= 0 ? line.substring(0,i) : line;
    }


    static void addFunctionLabelsLine(String line, Writer texWriter) 
        throws IOException {

        line = stripComments(line);
        
        Matcher matcher = FUNCTION_PATTERN.matcher(line);
        while (matcher.find()) {
            String functionName = matcher.group(1);
            texWriter.write("\\index{\\tt " + 
                            functionName.replaceAll("_","\\\\_")
                            + "}" + EOL);
        }
    }

    static void writeEnd(Writer texWriter)
        throws IOException {

        texWriter.write("\\printindex" + EOL);
        texWriter.write("\\end{document}" + EOL);
    }

}
