@Library('StanUtils')
import org.stan.Utils

def utils = new org.stan.Utils()
def skipRemainingStages = false
def skipOpenCL = false

def setupCXX(failOnError = true, CXX = CXX, String stanc3_bin_url = "nightly") {
    errorStr = failOnError ? "-Werror " : ""
    stanc3_bin_url_str = stanc3_bin_url != "nightly" ? "\nSTANC3_TEST_BIN_URL=${stanc3_bin_url}\n" : ""
    writeFile(file: "make/local", text: "CXX=${CXX} -Wno-inconsistent-missing-override ${errorStr}${stanc3_bin_url_str}")
}

def runTests(String testPath, Boolean separateMakeStep=true) {
    if (separateMakeStep) {
        sh "python3 runTests.py -j${PARALLEL} ${testPath} --make-only"
    }
    try { sh "python3 runTests.py -j${PARALLEL} ${testPath}" }
    finally { junit 'test/**/*.xml' }
}

def runTestsWin(String testPath, Boolean separateMakeStep=true) {
    withEnv(['PATH+TBB=./lib/stan_math/lib/tbb']) {
       if (separateMakeStep) {
           bat """
            SET \"PATH=C:\\Users\\jenkins\\Anaconda3;%PATH%\"
            SET \"PATH=${env.RTOOLS40_HOME};%PATH%\"
            SET \"PATH=${env.RTOOLS40_HOME}\\usr\\bin;${LLVM7}\\bin;%PATH%\" //
            SET \"PATH=${env.RTOOLS40_HOME}\\mingw64\\bin;%PATH%\"
            SET \"PATH=C:\\PROGRA~1\\R\\R-4.1.2\\bin;%PATH%\"
            SET \"PATH=C:\\PROGRA~1\\Microsoft^ MPI\\Bin;%PATH%\"
            SET \"MPI_HOME=C:\\PROGRA~1\\Microsoft^ MPI\\Bin\"
            python runTests.py -j${PARALLEL} ${testPath} --make-only
           """
       }
       try {
            bat """
                SET \"PATH=C:\\Users\\jenkins\\Anaconda3;%PATH%\"
                SET \"PATH=${env.RTOOLS40_HOME};%PATH%\"
                SET \"PATH=${env.RTOOLS40_HOME}\\usr\\bin;${LLVM7}\\bin;%PATH%\" //
                SET \"PATH=${env.RTOOLS40_HOME}\\mingw64\\bin;%PATH%\"
                SET \"PATH=C:\\PROGRA~1\\R\\R-4.1.2\\bin;%PATH%\"
                SET \"PATH=C:\\PROGRA~1\\Microsoft^ MPI\\Bin;%PATH%\"
                SET \"MPI_HOME=C:\\PROGRA~1\\Microsoft^ MPI\\Bin\"
                python runTests.py -j${PARALLEL} ${testPath}
            """
       }
       finally { junit 'test/**/*.xml' }
    }
}

def deleteDirWin() {
    bat "attrib -r -s /s /d"
    deleteDir()
}

String stanc3_bin_url() { params.stanc3_bin_url ?: "nightly" }
String cmdstan_pr() { params.cmdstan_pr ?: "downstream_tests" }
String stan_pr() {
    if (env.BRANCH_NAME == 'downstream_tests') {
        ''
    } else if (env.BRANCH_NAME == 'downstream_hotfix') {
        'master'
    } else {
        env.BRANCH_NAME
    }
}
String integration_tests_flags() {
    if (params.compile_all_model) {
        '--no-ignore-models '
    } else {
        ''
    }
}

def isBranch(String b) { env.BRANCH_NAME == b }
Boolean isPR() { env.CHANGE_URL != null }
String fork() { env.CHANGE_FORK ?: "stan-dev" }
String branchName() { isPR() ? env.CHANGE_BRANCH :env.BRANCH_NAME }

pipeline {
    agent none
    parameters {
        string(defaultValue: '', name: 'math_pr', description: "Leave blank "
                + "unless testing against a specific math repo pull request, "
                + "e.g. PR-640.")
        string(defaultValue: 'downstream_tests', name: 'cmdstan_pr',
          description: 'PR to test CmdStan upstream against e.g. PR-630')
        string(defaultValue: 'nightly', name: 'stanc3_bin_url',
          description: 'Custom stanc3 binary url')
        booleanParam(defaultValue: false, name: 'run_tests_all_os', description: 'Run unit and integration tests on all OS.')
        booleanParam(defaultValue: false, name: 'compile_all_models', description: 'Run integration tests on the full test model suite.')
    }
    options {
        skipDefaultCheckout()
        preserveStashes(buildCount: 7)
        parallelsAlwaysFailFast()
    }
    environment {
        GCC = 'g++'
        PARALLEL = 4
        MAC_CXX = 'clang++'
        LINUX_CXX = 'clang++-6.0'
        WIN_CXX = 'g++'
        GIT_AUTHOR_NAME = 'Stan Jenkins'
        GIT_AUTHOR_EMAIL = 'mc.stanislaw@gmail.com'
        GIT_COMMITTER_NAME = 'Stan Jenkins'
        GIT_COMMITTER_EMAIL = 'mc.stanislaw@gmail.com'
        OPENCL_DEVICE_ID_CPU = 0
        OPENCL_DEVICE_ID_GPU = 0
        OPENCL_PLATFORM_ID = 1
        OPENCL_PLATFORM_ID_CPU = 0
        OPENCL_PLATFORM_ID_GPU = 0
    }
    stages {

        stage('Kill previous builds') {
            when {
                not { branch 'develop' }
                not { branch 'master' }
                not { branch 'downstream_tests' }
            }
            steps {
                script {
                    utils.killOldBuilds()
                }
            }
        }
        stage("Clang-format") {
            agent {
                docker {
                    image 'stanorg/ci:gpu'
                    label 'linux'
                }
            }
            steps {
                retry(3) { checkout scm }
                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b',
                    usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    sh """#!/bin/bash
                        set -x
                        git checkout -b ${branchName()}
                        clang-format --version
                        find src -name '*.hpp' -o -name '*.cpp' | xargs -n20 -P${PARALLEL} clang-format -i
                        if [[ `git diff` != "" ]]; then
                            git config user.email "mc.stanislaw@gmail.com"
                            git config user.name "Stan Jenkins"
                            git add src
                            git commit -m "[Jenkins] auto-formatting by `clang-format --version`"
                            git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/${fork()}/stan.git ${branchName()}
                            echo "Exiting build because clang-format found changes."
                            echo "Those changes are now found on stan-dev/stan under branch ${branchName()}"
                            echo "Please 'git pull' before continuing to develop."
                            exit 1
                        fi
                    """
                }
            }
            post {
                always { deleteDir() }
                failure {
                    script {
                        emailext (
                            subject: "[StanJenkins] Autoformattted: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                            body: "Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' " +
                                "has been autoformatted and the changes committed " +
                                "to your branch, if permissions allowed." +
                                "Please pull these changes before continuing." +
                                "\n\n" +
                                "See https://github.com/stan-dev/stan/wiki/Coding-Style-and-Idioms" +
                                " for setting up the autoformatter locally.\n"+
                            "(Check console output at ${env.BUILD_URL})",
                            recipientProviders: [[$class: 'RequesterRecipientProvider']],
                            to: "${env.CHANGE_AUTHOR_EMAIL}"
                        )
                    }
                }
            }
        }
        stage('Linting & Doc checks') {
            agent {
                docker {
                    image 'stanorg/ci:gpu'
                    label 'linux'
                }
            }
            steps {
                script {
                    retry(3) { checkout scm }
                    sh """
                       make math-revert
                       make clean-all
                       git clean -xffd
                    """
                    utils.checkout_pr("math", "lib/stan_math", params.math_pr)
                    stash 'StanSetup'
                    setupCXX(true, LINUX_CXX)
                    parallel(
                        CppLint: { sh "make cpplint" },
                        API_docs: { sh 'make doxygen' },
                    )
                }
            }
            post {
                always {

                    recordIssues id: "lint_doc_checks",
                    name: "Linting & Doc checks",
                    enabledForFailure: true,
                    aggregatingResults : true,
                    tools: [
                        cppLint(id: "cpplint", name: "Linting & Doc checks@CPPLINT")
                    ],
                    blameDisabled: false,
                    qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                    healthy: 10, unhealthy: 100, minimumSeverity: 'HIGH',
                    referenceJobName: env.BRANCH_NAME

                    deleteDir()
                }
            }
        }
        stage('Verify changes') {
            agent {
                docker {
                    image 'stanorg/ci:gpu'
                    label 'linux'
                }
            }
            steps {
                script {

                    retry(3) { checkout scm }
                    sh 'git clean -xffd'

                    // These paths will be passed to git diff
                    // If there are changes to them, CI/CD will continue else skip
                    def paths = ['make', 'src/stan', 'src/test', 'Jenkinsfile', 'makefile', 'runTests.py',
                        'lib/stan_math/stan', 'lib/stan_math/make', 'lib/stan_math/lib', 'lib/stan_math/test',
                        'lib/stan_math/runTests.py', 'lib/stan_math/runChecks.py', 'lib/stan_math/makefile',
                        'lib/stan_math/Jenkinsfile', 'lib/stan_math/.clang-format'
                    ].join(" ")

                    skipRemainingStages = utils.verifyChanges(paths)

                    def openCLPaths = ['src/stan/model/indexing'].join(" ")
                    skipOpenCL = utils.verifyChanges(openCLPaths)
                }
            }
            post {
                always {
                    deleteDir()
                }
            }
        }
        stage('Unit tests') {
            when {
                expression {
                    !skipRemainingStages
                }
            }
            parallel {
                stage('Windows Headers & Unit') {
                    agent { label 'windows' }
                    when {
                        expression {
                            !skipRemainingStages
                        }
                    }
                    steps {
                        deleteDirWin()
                            unstash 'StanSetup'
                            bat """
                                SET \"PATH=${env.RTOOLS40_HOME};%PATH%\"
                                SET \"PATH=${env.RTOOLS40_HOME}\\usr\\bin;${LLVM7}\\bin;%PATH%\" //
                                SET \"PATH=${env.RTOOLS40_HOME}\\mingw64\\bin;%PATH%\"
                                SET \"PATH=C:\\PROGRA~1\\R\\R-4.1.2\\bin;%PATH%\"
                                SET \"PATH=C:\\PROGRA~1\\Microsoft^ MPI\\Bin;%PATH%\"
                                SET \"MPI_HOME=C:\\PROGRA~1\\Microsoft^ MPI\\Bin\"
                                mingw32-make.exe -f lib/stan_math/make/standalone math-libs
                                mingw32-make.exe -j${PARALLEL} test-headers
                            """
                            setupCXX(false, WIN_CXX, stanc3_bin_url())
                            runTestsWin("src/test/unit")
                    }
                    post { always { deleteDirWin() } }
                }
                stage('Linux Unit') {
                    agent {
                        docker {
                            image 'stanorg/ci:gpu'
                            label 'linux'
                            args '--pull always --gpus 1'
                        }
                    }
                    steps {
                        unstash 'StanSetup'
                        setupCXX(true, LINUX_CXX, stanc3_bin_url())
                        sh """
                            echo STAN_OPENCL=true > make/local
                            echo OPENCL_PLATFORM_ID=${OPENCL_PLATFORM_ID_GPU} >> make/local
                            echo OPENCL_DEVICE_ID=${OPENCL_DEVICE_ID_GPU} >> make/local
                        """
                        sh """
                            make -j${PARALLEL} test-headers
                        """
                        runTests("src/test/unit")
                    }
                    post { always { deleteDir() } }
                }
                stage('Mac Unit') {
                agent { label 'osx' }
                    when {
                        expression {
                            ( env.BRANCH_NAME == "develop" ||
                            env.BRANCH_NAME == "master" ||
                            params.run_tests_all_os ) &&
                            !skipRemainingStages
                        }
                    }
                    steps {
                        unstash 'StanSetup'
                        setupCXX(false, MAC_CXX, stanc3_bin_url())
                        runTests("src/test/unit")
                    }
                    post { always { deleteDir() } }
                }
            }
        }
        stage('Integration') {
            when {
                expression {
                    !skipRemainingStages
                }
            }
            parallel {
                stage('Integration Linux') {
                    agent {
                        docker {
                            image 'stanorg/ci:gpu'
                            label 'linux'
                        }
                    }
                    steps {
                        sh """
                            git clone --recursive https://github.com/stan-dev/performance-tests-cmdstan
                            git clone https://github.com/stan-dev/stanc3/ performance-tests-cmdstan/stanc3
                        """
                        script {
                            if (params.cmdstan_pr != 'downstream_tests') {
                                if(params.cmdstan_pr.contains("PR-")){
                                    pr_number = params.cmdstan_pr.split("-")[1]
                                    sh """
                                        cd performance-tests-cmdstan/cmdstan
                                        git fetch origin pull/${pr_number}/head:pr/${pr_number}
                                        git checkout pr/${pr_number}
                                    """
                                }else{
                                    sh """
                                        cd performance-tests-cmdstan/cmdstan
                                        git checkout develop && git pull && git checkout ${params.cmdstan_pr}
                                    """
                                }
                            }
                            if (params.stanc3_bin_url != 'nightly') {
                                sh """
                                    cd performance-tests-cmdstan/cmdstan
                                    echo 'STANC3_TEST_BIN_URL=${params.stanc3_bin_url}' >> make/local
                                """
                            }
                        }
                        dir('performance-tests-cmdstan/cmdstan/stan'){
                            unstash 'StanSetup'
                            script {
                                if (params.stanc3_bin_url != 'nightly') {
                                    sh """
                                        echo 'STANC3_TEST_BIN_URL=${params.stanc3_bin_url}' >> make/local
                                    """
                                }
                            }
                        }
                        sh """
                            cd performance-tests-cmdstan/cmdstan
                            echo 'O=0' >> make/local
                            echo 'CXX=${LINUX_CXX}' >> make/local
                            make clean-all
                            make -j${PARALLEL} build
                            cd ..
                            python3 ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 stanc3/test/integration/good
                            python3 ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 example-models
                        """
                        sh """
                            cd performance-tests-cmdstan/cmdstan/stan
                            python3 ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            python3 ./runTests.py src/test/integration/standalone_functions_test.cpp
                            python3 ./runTests.py src/test/integration/multiple_translation_units_test.cpp
                        """
                    }
                    post { always { deleteDir() } }
                }
                stage('Integration Mac') {
                    agent { label 'osx' }
                    when {
                        expression {
                            ( env.BRANCH_NAME == "develop" ||
                            env.BRANCH_NAME == "master" ||
                            params.run_tests_all_os ) &&
                            !skipRemainingStages
                        }
                    }
                    steps {
                        sh """
                            git clone --recursive https://github.com/stan-dev/performance-tests-cmdstan
                        """
                        dir('performance-tests-cmdstan/cmdstan/stan'){
                            unstash 'StanSetup'
                        }
                        sh """
                            cd performance-tests-cmdstan/cmdstan
                            echo 'O=0' >> make/local
                            echo 'CXX=${MAC_CXX}' >> make/local
                            make clean-all
                            make -j${PARALLEL} build
                            cd ..
                            python3 ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 stanc3/test/integration/good
                            python3 ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 example-models
                        """
                        sh """
                            cd performance-tests-cmdstan/cmdstan/stan
                            python3 ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            python3 ./runTests.py src/test/integration/standalone_functions_test.cpp
                            python3 ./runTests.py src/test/integration/multiple_translation_units_test.cpp
                        """
                    }
                    post { always { deleteDir() } }
                }
                stage('Integration Windows') {
                    agent { label 'windows' }
                    when {
                        expression {
                            !skipRemainingStages
                        }
                    }
                    steps {
                        deleteDirWin()
                        bat """
                            git clone --recursive https://github.com/stan-dev/performance-tests-cmdstan
                        """
                        dir('performance-tests-cmdstan/cmdstan/stan'){
                            unstash 'StanSetup'
                        }
                        writeFile(file: "performance-tests-cmdstan/cmdstan/make/local", text: "CXX=${WIN_CXX}\nPRECOMPILED_HEADERS=true")
                        withEnv(["PATH+TBB=${WORKSPACE}\\performance-tests-cmdstan\\cmdstan\\stan\\lib\\stan_math\\lib\\tbb"]) {

                            bat """
                                SET \"PATH=C:\\Users\\jenkins\\Anaconda3;%PATH%\"
                                SET \"PATH=${env.RTOOLS40_HOME};%PATH%\"
                                SET \"PATH=${env.RTOOLS40_HOME}\\usr\\bin;${LLVM7}\\bin;%PATH%\" //
                                SET \"PATH=${env.RTOOLS40_HOME}\\mingw64\\bin;%PATH%\"
                                SET \"PATH=C:\\PROGRA~1\\R\\R-4.1.2\\bin;%PATH%\"
                                SET \"PATH=C:\\PROGRA~1\\Microsoft^ MPI\\Bin;%PATH%\"
                                SET \"MPI_HOME=C:\\PROGRA~1\\Microsoft^ MPI\\Bin\"
                                cd performance-tests-cmdstan/cmdstan
                                mingw32-make.exe -j${PARALLEL} build
                                cd ..
                                python ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 stanc3/test/integration/good
                                python ./runPerformanceTests.py -j${PARALLEL} ${integration_tests_flags()}--runs=0 example-models
                            """
                        }
                        bat """
                            SET \"PATH=C:\\Users\\jenkins\\Anaconda3;%PATH%\"
                            SET \"PATH=${env.RTOOLS40_HOME};%PATH%\"
                            SET \"PATH=${env.RTOOLS40_HOME}\\usr\\bin;${LLVM7}\\bin;%PATH%\" //
                            SET \"PATH=${env.RTOOLS40_HOME}\\mingw64\\bin;%PATH%\"
                            SET \"PATH=C:\\PROGRA~1\\R\\R-4.1.2\\bin;%PATH%\"
                            SET \"PATH=C:\\PROGRA~1\\Microsoft^ MPI\\Bin;%PATH%\"
                            SET \"MPI_HOME=C:\\PROGRA~1\\Microsoft^ MPI\\Bin\"
                            cd performance-tests-cmdstan/cmdstan/stan
                            python ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            python ./runTests.py src/test/integration/standalone_functions_test.cpp
                            python ./runTests.py src/test/integration/multiple_translation_units_test.cpp
                        """
                    }
                    post { always { deleteDirWin() } }
                }
            }
        }
        stage('Upstream CmdStan tests') {
            when {
                    expression {
                        ( env.BRANCH_NAME ==~ /PR-\d+/ ||
                        env.BRANCH_NAME == "downstream_tests" ||
                        env.BRANCH_NAME == "downstream_hotfix" ) &&
                        !skipRemainingStages
                    }
                }
            steps {
                build(job: "Stan/CmdStan/${cmdstan_pr()}",
                      parameters: [
                        string(name: 'stan_pr', value: stan_pr()),
                        string(name: 'math_pr', value: params.math_pr),
                        string(name: 'stanc3_bin_url', value: stanc3_bin_url())
                      ])
            }
        }

    }
    // Below lines are commented to avoid spamming emails during migration/debug
    post {
        always {
            node("linux") {
                recordIssues id: "pipeline",
                name: "Entire pipeline results",
                enabledForFailure: true,
                aggregatingResults : false,
                filters: [
                    excludeFile('lib/.*')
                ],
                tools: [
                    gcc4(id: "pipeline_gcc4", name: "GNU C Compiler"),
                    clang(id: "pipeline_clang", name: "LLVM/Clang")
                ],
                blameDisabled: false,
                qualityGates: [[threshold: 30, type: 'TOTAL', unstable: true]],
                healthy: 10, unhealthy: 100, minimumSeverity: 'HIGH',
                referenceJobName: env.BRANCH_NAME
            }
        }
        success {
            script {
                utils.updateUpstream(env,'cmdstan')
                utils.mailBuildResults("SUCCESSFUL")
            }
        }
        unstable { script { utils.mailBuildResults("UNSTABLE", "stan-buildbot@googlegroups.com") } }
        failure { script { utils.mailBuildResults("FAILURE", "stan-buildbot@googlegroups.com") } }
    }
}
