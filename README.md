# AquaSense

AquaSense translates a StormIR or Stan probabilistic program to *PyTorch* code for the purpose of sensitivity analysis

# Installation

**Prerequisites:**

* Java 

  Check installation with `java -version`. If not installed, on Ubuntu you may try `sudo apt -y update; sudo apt install openjdk-8-jdk`

  Make sure the location of your java binary is added to PATH in your command-line environment, as `java ...` is used by AquaSense's helper scripts

* Maven

  Check installation with `mvn --version`. If not installed, on Ubuntu you may try `sudo apt -y update; sudo apt install maven`
  
* *PyTorch* (for running the generated code)

  To install PyTorch with your version of CUDA and preferred package manager, check out https://pytorch.org/get-started/locally/


**Install dependencies and build AquaSense:**

In the root directory of this repo, run
    
    mvn package -DskipTests=true

For Apple silicon users, run 

    mvn package -DskipTests=true -U -Djavacpp.platform=macosx-x86_64

Blocked mirror link error [refer](https://stackoverflow.com/questions/67833372/getting-blocked-mirror-for-repositories-maven-error-even-after-adding-mirrors/67835542#67835542)
Refer to the following for other installation-related errors specific to Apple Silicon [link](https://community.konduit.ai/t/support-for-apple-silicon-m1/1168)

In the end it should print `BUILD SUCCESS`.

# Quick Demo

## 1) Run AquaSense on all benchmarks
```
./benchmark.py
```

## 2) Analyze the sensitivity of a parameter in the model [neural](benchmarks/stan_bench/neural/neural.stan)

```
./aquasense.py benchmarks/stan_bench/neural
```

# Usage

## 1) Easy sensitivity analysis with `aquasense.py`

**We provide a helper script `aquasense.py` to translate, run, and visualize the sensitivity of a probabilistic program**

The script automatically increases the granularity of quantization (think of it as a accuracy hyperparameter) to approximate the true sensitivity, until the approximation is deemed to have converged; the results are shown in a plot

**Usage:**

```
./aquasense.py <path_to_model_dir> [-h] [-v RANDVAR] [-p PARAMETER] [-b BOUNDS] [-s SPLITS] [-m M] [-c C]
```
**E.g.**
```
./aquasense.py benchmarks/stan_bench/neural
```

**E.g.**
```
./aquasense.py benchmarks/stan_bench/neural -v "w[0]" -p 0 -b -0.1 0.1 -m expdist1
```

**E.g.** the sensitivity interpolations using different #splits

![neural results](neural.png)

Or alternatively, one can manually translate the probabilistic program, then perform sensitivity analysis, see 2) and 3)
## 2) Translate probabilistic programs to *PyTorch*

AquaSense works as a source-to-source translator that takes as input either  
- a program in [Storm IR](https://misailo.cs.illinois.edu/papers/storm-fse19.pdf) (`<prog_name>.template`). Example: benchmarks/psense_bench/coins/coins.template
- a directory `<prog_name>/` containing [Stan](https://mc-stan.org/) file (`<prog_name>.stan`) and data (`<prog_name>.data.R`). Example: benchmarks/stan_bench/neural/

and output a python script, such as, benchmarks/stan_bench/neural/neural.py, to be used for sensitivity analysis

#### (a) Generate torch from a Storm IR file
**Usage:**
    
    java -cp "target/aqua-1.0.jar:lib/storm-1.0.jar" aqua.analyses.PyCompilerRunner <path_to_input_template_file>
    
**E.g.:**

    java -cp "target/aqua-1.0.jar:lib/storm-1.0.jar" aqua.analyses.PyCompilerRunner ./benchmarks/psense_bench/coins/coins.template
    

#### (b) Generate torch from a Stan file
The `path_to_input_dir` must contain a stan file (`<prog_name>.stan`) and a data file (`<prog_name>.data.R`) with the same name as the directory.

**Usage:**
    
    java -cp "target/aqua-1.0.jar:lib/storm-1.0.jar" aqua.analyses.PyCompilerRunner <path_to_input_dir>
    
**E.g.:**

    java -cp "target/aqua-1.0.jar:lib/storm-1.0.jar" aqua.analyses.PyCompilerRunner ./benchmarks/stan_bench/anova_radon_nopred

 The directory `./benchmarks/stan_bench/anova_radon_nopred` contains `anova_radon_nopred.stan` and `anova_radon_nopred.data.R`.


## 3) Running the *PyTorch* code directly

For each probabilistic program, there will be a `<prog_name>.py` containing the translated Pytorch code. It is under the same directory as the input `<prog_name>.template` or `<prog_name>.stan` file.

By default, AquaSense uses GPU for tensor computations in *PyTorch*, and analyzes the sensitivity of the first parameter of the first identified random variable within a range.

E.g.
```
python3 benchmarks/stan_bench/neural/neural.py
```

To specify a random variable, use the option `-v`; to specify the parameter (index), use the option `-p`; to specify the noise bound, use the option `-b`. e.g.

E.g. this command analyzes the sensitivity of parameter 0 (lower bound) of the random variable "w[0]" with noise in between -0.1 and 0.1
```
python3 benchmarks/stan_bench/neural/neural.py -v "w[0]" -p 0 -b -0.1 0.1
```

For more details on the usage, use the `-h` option

## Notes
* AquaSense is shown empiricially to be exact on discrete probabilistic models, therefore there is no need to approximate the true sensitivity of models like [coins](benchmarks/psense_bench/coins/coins.template) using the helper script. One can simply perform step 2) and 3) and observe/visualize the outputs
* On certain discrete models, the injection of arbitrary noise into parameters can make certain distributions ill-defined, e.g. `UniformInt(0.314, 5)`, leading to numerical issues. It is recommended to choose the noise vector manually.

# Project Structure

        .  
        ├── benchmarks/                                         # All benchmarks
        │     ├── stan_bench/                                   # Benchmarks in Stan
        │     └── psense_bench/                                 # Benchmarks in Storm IR
        │
        ├── src/                                                # AQUA source code in Java
        │     ├── main/                  
        │     │     ├── java/                       
        │     │     │   └──  aqua/ 
        │     │     │        ├── analyses/                      # AQUA Analysis code
        │     │     │        │    ├── PyCompilerRunner.java     # Program entry point. Translates file, constructs CFG, and run compiler
        │     │     │        │    ├── PytorchCompiler.java      # Generate Pytorch code
        │     │     │        │    └── PytorchVisitor.java       # Used in PytorchCompiler to generates code for statements and expressions
        │     │     │        └── cfg/CFGBuilder.java            # CFG constructor for Storm IR
        │     │     └── resources/                              # Json files for properties of distributions and the config of Storm IR
        │     └── test/java/tests/                              # Unit tests in the development
        │ 
        ├── lib/grammar-1.0.jar                                 # Storm IR jar 
        ├── aquasense.py                                        # Helper script to translate, run and visualize 
        ├── converge.py                                         # Define convergence criteria
        ├── metrics.py                                          # Define distance metrics
        |
        ├── README.md                                           # README for basic info  
        ├── antlr-4.7.1-complete.jar                            # ANTLR jar used for parsing Stan / Storm IR files
        └── pom.xml                                             # POM file in maven for project configuration and dependency

