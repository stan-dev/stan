def props = [
  buildDiscarder(logRotator(numToKeepStr: '20', daysToKeepStr: '30')),
  parameters([
    string(defaultValue: '', name: 'math_pr', description: "Leave blank "
            + "unless testing against a specific math repo pull request, "
            + "e.g. PR-640."),
    string(defaultValue: 'develop', name: 'cmdstan_pr',
      description: 'PR to test CmdStan upstream against e.g. PR-630'),
    string(defaultValue: 'nightly', name: 'stanc3_bin_url',
      description: 'Custom stanc3 binary url'),
    booleanParam(defaultValue: false, name: 'downsteam', description: 'Run downstream tests from math (was previously downstream_hotfix/downstream_tests [develop])'),
    booleanParam(defaultValue: false, name: 'run_tests_all_os', description: 'Run unit and integration tests on all OS.'),
    booleanParam(defaultValue: false, name: 'compile_all_models', description: 'Run integration tests on the full test model suite.'),
    booleanParam(defaultValue: false, name: 'run_all', description: 'Pretend all files changes'),
  ])
]

if (!params.downstream) {
  props <<= disableConcurrentBuilds()
}

properties(props)

def image = 'stanorg/ci:v1'
def commit
def runRemainingStages = false
def LINUX_CXX = 'clang++-7 -Werror -Wno-inconsistent-missing-override -Wno-error=return-type -Wno-error=division-by-zero'
def WIN_CXX = 'g++ -Werror -Wno-error=overloaded-virtual -Wno-error=template-id-cdtor -Wno-error=deprecated-declarations -Wno-error=cast-user-defined -Wno-error=unused-value -Wno-error=array-bounds'
def MAC_CXX = 'clang++' // -Werror -Wno-inconsistent-missing-override -Wno-unused-but-set-variable
def WINSETENV = '''
  SET "PATH=%RTOOLS%\\x86_64-w64-mingw32.static.posix\\bin;%RTOOLS%;%RTOOLS%\\usr\\bin;%CONDA%;%PATH%"
'''
def stanc3_bin_url = params.stanc3_bin_url != "nightly" ? "STANC3_TEST_BIN_URL=${params.stanc3_bin_url}\n" : ''

catchError {
  withEnv([
    'GCC=g++',
    'GIT_AUTHOR_NAME=Stan Jenkins',
    'GIT_AUTHOR_EMAIL=mc.stanislaw@gmail.com',
    'GIT_COMMITTER_NAME=Stan Jenkins',
    'GIT_COMMITTER_EMAIL=mc.stanislaw@gmail.com'
  ]) {
    runPod(image: image, cpus: 2) {
      stage('Verify changes') {
        commit = sh(returnStdout: true, script: "git rev-parse HEAD").trim()
        runRemainingStages = params.downstream || params.run_all || filesChanged(
          'make', 'src/stan', 'src/test', 'Jenkinsfile', 'makefile', 'runTests.py',
          'lib/stan_math/stan', 'lib/stan_math/make', 'lib/stan_math/lib', 'lib/stan_math/test',
          'lib/stan_math/runTests.py', 'lib/stan_math/runChecks.py', 'lib/stan_math/makefile',
          'lib/stan_math/Jenkinsfile', 'lib/stan_math/.clang-format')
      }

      stage("Clang-format") {
        def dirty = sh returnStatus: true, script: """
          clang-format --version
          git ls-files 'src/*.hpp' 'src/*.cpp' | xargs -n20 -P\$PARALLEL clang-format -i
          git diff --exit-code
        """
        if (dirty) {
          def branch = env.CHANGE_BRANCH ?: env.BRANCH_NAME
          def repo = env.CHANGE_FORK ?: "stan-dev"
          if (!("/" in repo))
            repo += "/stan.git"
          echo "Exiting build because clang-format found changes."
          emailext (
              subject: "[StanJenkins] Autoformattted: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
              body: """
Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' has been autoformatted and the
changes committed to your branch, if permissions allowed.  Please pull these
changes before continuing.

See https://github.com/stan-dev/stan/wiki/Coding-Style-and-Idioms for setting
up the autoformatter locally.  (Check console output at ${env.BUILD_URL})
""",
              recipientProviders: [[$class: 'RequesterRecipientProvider']],
              to: env.CHANGE_AUTHOR_EMAIL)
          sh '''
            git add -u src
            git commit -m "[Jenkins] auto-formatting by `clang-format --version`"
          '''
          gitPush(gitScm: scmGit(
              userRemoteConfigs: [[credentialsId: "stan-github", name: 'dest', url: "https://github.com/$repo"]],
              branches: [[name: "refs/heads/$branch"]]),
              targetBranch: branch,
              targetRepo: 'dest')
          echo "Those changes are now found on stan-dev/stan under $repo branch $branch"
          echo "Please 'git pull' before continuing to develop."
          error "clang-format changes"
        }
      }

      stage('Linting & Doc checks') {
        checkoutPR("lib/stan_math", params.math_pr)
        writeFile(file: "make/local", text: "CXX=$LINUX_CXX\n$stanc3_bin_url")
        parallel(
          CppLint: { sh "make cpplint" },
          API_docs: { sh 'make doxygen' },
        )
      }
    }

    if (runRemainingStages) {
      stage('Unit tests') {
        def runUnit = { args ->
          def local = "CXX=$args.cxx\n$stanc3_bin_url"
          if (args.local)
            local += args.local
          writeFile(file: "make/local", text: local)
          def pre = args.pre ?: ''
          batsh(pre + 'make -j$PARALLEL test-headers')
          batsh(pre + 'python3 runTests.py -j$PARALLEL src/test/unit --make-only')
          catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
            batsh(pre + 'python3 runTests.py -j$PARALLEL src/test/unit')
          }
          junit 'test/**/*.xml'
        }

        parallel windows: {
          node('windows') {
            stage('Windows Headers & Unit') {
              checkout scm
              bat """$WINSETENV
                  make -f lib/stan_math/make/standalone math-libs
              """
              withEnv(["PATH+TBB=$WORKSPACE\\lib\\stan_math\\lib\\tbb"]) {
                runUnit(cxx: WIN_CXX, pre: WINSETENV)
              }
            }
          }
        }, linux: {
          runPod(image: image, gpus: 1) {
            stage('Linux Unit') {
              runUnit(cxx: LINUX_CXX, local: """
STAN_OPENCL=true
OPENCL_PLATFORM_ID=0
OPENCL_DEVICE_ID=0
LDFLAGS_OPENCL=-L/usr/local/cuda/targets/x86_64-linux/lib
""")
            }
          }
        }, mac: {
          if (!params.downstream && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master") || params.run_tests_all_os) {
            node('macos') {
              stage('Mac Unit') {
                checkout scm
                runUnit(cxx: MAC_CXX)
              }
            }
          }
        }
      }

      stage('Integration') {
        // TODO: this was disabled before
        def integration_tests_flags = params.compile_all_models ? '--no-ignore-models' : ''
        def runIntegration = { args ->
          def pre = args.pre ?: ''
          deleteDir()
          checkout scmGit(userRemoteConfigs: [[url: 'https://github.com/stan-dev/performance-tests-cmdstan']],
            extensions: [cloneOption(shallow: true, depth: 2), submodule(recursiveSubmodules: true, shallow: true, depth: 32)])
          dir('stanc3') {
            checkout scmGit(userRemoteConfigs: [[url: 'https://github.com/stan-dev/stanc3']],
              extensions: [cloneOption(shallow: true, depth: 8)])
          }
          checkoutPR('cmdstan', params.cmdstan_pr)
          dir('cmdstan/stan') {
            checkout scm
            writeFile(file: 'make/local', text: stanc3_bin_url)
          }
          writeFile(file: 'cmdstan/make/local', text: args.local+"\n$stanc3_bin_url")
          batsh pre + """
            make -C cmdstan -j\$PARALLEL build
            python3 ./runPerformanceTests.py -j\$PARALLEL $integration_tests_flags --runs=0 stanc3/test/integration/good
            python3 ./runPerformanceTests.py -j\$PARALLEL $integration_tests_flags --runs=0 example-models
          """
          dir('cmdstan/stan') {
            batsh pre + """
                python3 ./runTests.py src/test/integration/compile_standalone_functions_test.cpp
                python3 ./runTests.py src/test/integration/standalone_functions_test.cpp
                python3 ./runTests.py src/test/integration/multiple_translation_units_test.cpp
            """
          }
        }

        parallel linux: {
          runPod(image: image, checkout: false, cpus: 16, memory: '128Gi') {
            stage('Integration Linux') {
              runIntegration(local: "O=0\nCXX=${LINUX_CXX}")
            }
          }
        }, mac: {
          if (!params.downstream && (env.BRANCH_NAME == 'develop' || env.BRANCH_NAME == 'master') || params.run_tests_all_os) {
            node('macos') {
              stage('Integration Mac') {
                runIntegration(local: "O=0\nCXX=${MAC_CXX}")
              }
            }
          }
        }, windows: {
          node('windows') {
            stage('Integration Windows') {
              withEnv(["PATH+TBB=${WORKSPACE}\\cmdstan\\stan\\lib\\stan_math\\lib\\tbb"]) {
                runIntegration(local: "CXX=${WIN_CXX}\nPRECOMPILED_HEADERS=true\n", pre: WINSETENV)
              }
            }
          }
        }
      }

      if (env.CHANGE_TARGET || params.downstream) {
        stage('Upstream CmdStan tests') {
          build(job: "CCM/Stan/cmdstan/$params.cmdstan_pr",
            parameters: [
              booleanParam(name: 'downstream', value: true),
              string(name: 'stan_pr', value: env.BRANCH_NAME),
              string(name: 'math_pr', value: params.math_pr),
              string(name: 'stanc3_bin_url', value: params.stanc3_bin_url)
            ])
        }
      }
    }
  }

  if (env.BRANCH_NAME == 'develop') {
    podTemplate(inheritFrom: 'jnlp') {
      node(POD_LABEL) {
        stage('Update upstream') {
          dir('cmdstan') {
            updateSubmodule('cmdstan', 'develop', 'stan', commit)
          }
          dir('rstan') {
            updateSubmodule('rstan', 'develop', 'StanHeaders/inst/include/upstream', commit)
          }
        }
      }
    }
  }
}

emailFailure()
