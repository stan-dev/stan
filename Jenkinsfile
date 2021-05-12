@Library('StanUtils')
import org.stan.Utils

def utils = new org.stan.Utils()
def skipRemainingStages = false

def setupCXX(failOnError = true, CXX = env.CXX, String stanc3_bin_url = "nightly") {
    errorStr = failOnError ? "-Werror " : ""
    stanc3_bin_url_str = stanc3_bin_url != "nightly" ? "\nSTANC3_TEST_BIN_URL=${stanc3_bin_url}\n" : ""
    writeFile(file: "make/local", text: "CXX=${CXX} ${errorStr}${stanc3_bin_url_str}")
}

def runTests(String testPath, Boolean separateMakeStep=true) {
    if (separateMakeStep) {
        sh "./runTests.py -j${env.PARALLEL} ${testPath} --make-only"
    }
    try { sh "./runTests.py -j${env.PARALLEL} ${testPath}" }
    finally { junit 'test/**/*.xml' }
}

def runTestsWin(String testPath, Boolean separateMakeStep=true) {
    withEnv(['PATH+TBB=./lib/stan_math/lib/tbb']) {
       if (separateMakeStep) {
           bat "runTests.py -j${env.PARALLEL} ${testPath} --make-only"
       }
       try { bat "runTests.py -j${env.PARALLEL} ${testPath}" }
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
            agent any
            steps {
                sh "printenv"
                deleteDir()
                retry(3) { checkout scm }
                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b',
                    usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    sh """#!/bin/bash
                        set -x
                        git checkout -b ${branchName()}
                        clang-format --version
                        find src -name '*.hpp' -o -name '*.cpp' | xargs -n20 -P${env.PARALLEL} clang-format -i
                        if [[ `git diff` != "" ]]; then
                            git config --global user.email "mc.stanislaw@gmail.com"
                            git config --global user.name "Stan Jenkins"
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
            agent any
            steps {
                script {
                    sh "printenv"
                    retry(3) { checkout scm }
                    sh """
                       make math-revert
                       make clean-all
                       git clean -xffd
                    """
                    utils.checkout_pr("math", "lib/stan_math", params.math_pr)
                    stash 'StanSetup'
                    setupCXX(true, env.GCC)
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
            agent { label 'linux' }
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
                            ( env.BRANCH_NAME == "develop" ||
                            env.BRANCH_NAME == "master" ||
                            params.run_tests_all_os ) &&
                            !skipRemainingStages
                        }
                    }
                    steps {
                        deleteDirWin()
                            unstash 'StanSetup'
                            bat "mingw32-make -f lib/stan_math/make/standalone math-libs"
                            bat "mingw32-make -j${env.PARALLEL} test-headers"
                            setupCXX(false, env.CXX, stanc3_bin_url())
                            runTestsWin("src/test/unit")
                    }
                    post { always { deleteDirWin() } }
                }
                stage('Linux Unit') {
                    agent { label 'linux' }
                    steps {
                        unstash 'StanSetup'
                        setupCXX(true, env.GCC, stanc3_bin_url())
                        sh "make -j${env.PARALLEL} test-headers"
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
                        setupCXX(false, env.CXX, stanc3_bin_url())
                        runTests("src/test/unit")
                    }
                    post { always { deleteDir() } }
                }
            }
        }
        stage('Integration') {
            parallel {
                stage('Integration Linux') {
                    agent { label 'linux' }
                    steps {
                        sh """
                            git clone --recursive https://github.com/stan-dev/performance-tests-cmdstan
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
                            echo 'CXX=${env.CXX}' >> make/local
                            make -j${env.PARALLEL} build
                            cd ..
                            ./runPerformanceTests.py -j${env.PARALLEL} ${integration_tests_flags()}--runs=0 cmdstan/stan/src/test/test-models/good
                        """
                        sh """
                            cd performance-tests-cmdstan/cmdstan/stan
                            ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            ./runTests.py src/test/integration/standalone_functions_test.cpp
                            ./runTests.py src/test/integration/multiple_translation_units_test.cpp
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
                            echo 'CXX=${env.CXX}' >> make/local
                            make -j${env.PARALLEL} build
                            cd ..
                            ./runPerformanceTests.py -j${env.PARALLEL} ${integration_tests_flags()}--runs=0 cmdstan/stan/src/test/test-models/good
                        """
                        sh """
                            cd performance-tests-cmdstan/cmdstan/stan
                            ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            ./runTests.py src/test/integration/standalone_functions_test.cpp
                            ./runTests.py src/test/integration/multiple_translation_units_test.cpp
                        """
                    }
                    post { always { deleteDir() } }
                }
                stage('Integration Windows') {
                    agent { label 'windows-ec2' }
                    when {
                        expression {
                            ( env.BRANCH_NAME == "develop" ||
                            env.BRANCH_NAME == "master" ||
                            params.run_tests_all_os ) &&
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
                        writeFile(file: "performance-tests-cmdstan/cmdstan/make/local", text: "CXX=${CXX}\nPRECOMPILED_HEADERS=true")
                        withEnv(["PATH+TBB=${WORKSPACE}\\performance-tests-cmdstan\\cmdstan\\stan\\lib\\stan_math\\lib\\tbb"]) {  
                            
                            bat """
                                cd performance-tests-cmdstan/cmdstan
                                mingw32-make -j${env.PARALLEL} build
                                cd ..
                                python ./runPerformanceTests.py -j${env.PARALLEL} ${integration_tests_flags()}--runs=0 cmdstan/stan/src/test/test-models/good
                            """
                        }
                        bat """
                            cd performance-tests-cmdstan/cmdstan/stan
                            python ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                            python ./runTests.py src/test/integration/standalone_functions_test.cpp
                            python ./runTests.py src/test/integration/multiple_translation_units_test.cpp
                        """
                    }
                    post { always { deleteDirWin() } }
                }
            }
            when {
                expression {
                    !skipRemainingStages
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
                build(job: "CmdStan/${cmdstan_pr()}",
                      parameters: [string(name: 'stan_pr', value: stan_pr()),
                                   string(name: 'math_pr', value: params.math_pr)])
            }
        }
        stage('Performance') {
            when {
                expression {
                    !skipRemainingStages
                }
            }
            agent { label 'oldimac' }
            steps {
                unstash 'StanSetup'
                setupCXX(true, env.CXX, stanc3_bin_url())
                sh """
                    ./runTests.py -j${env.PARALLEL} src/test/performance
                    cd test/performance
                    RScript ../../src/test/performance/plot_performance.R
                """
            }
            post {
                always {
                    retry(2) {
                        junit 'test/**/*.xml'
                        archiveArtifacts 'test/performance/performance.csv,test/performance/performance.png'
                        perfReport compareBuildPrevious: true, errorFailedThreshold: 0, errorUnstableThreshold: 0, failBuildIfNoResultFile: false, modePerformancePerTestCase: true, sourceDataFiles: 'test/performance/**.xml'
                    }
                    deleteDir()
                }
            }
        }
    }
    post {
        always {
            node("osx || linux") {
                recordIssues id: "pipeline",
                name: "Entire pipeline results",
                enabledForFailure: true,
                aggregatingResults : false,
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
