package aqua.analyses;

import aqua.cfg.CFGBuilder;
import grammar.AST;
import grammar.analyses.Pair;
import grammar.cfg.BasicBlock;
import grammar.cfg.Section;
import grammar.cfg.SectionType;
import grammar.cfg.Statement;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.ArrayUtils;
import translators.Stan2IRTranslator;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("FieldMayBeFinal")
public class PytorchCompiler {
    // config
    private static boolean sameInnerSplits; //= true;
    private static boolean adaptive; // = false; // use adaptive algo even if all params are given bounds
    private static boolean enableTiming; // = false;
    private static boolean getCFGPlot; // = false;
    private static boolean getDataGrad; // = false;
    private static String splits; // = "50";
    private static Properties properties = new Properties();

    // sense begins ---------------------------------------------------------------------------------------------
    // the index of the parameter (that will be instrumented with noise)
    private int paramIndex = 0;

    private Set<String> otherDists = new HashSet<>(Arrays.asList("geometric", "laplace", "cauchy"));

    // map param name to its parameter indices, vars are stored in the order of declaration
    // e.g. p2ni['uniformrv2'] = List(3,4)
    private Map<String, List<Integer>> var2NoiseIndices = new LinkedHashMap<>();

    // store the num of params of each var that has been instrumented with noise
    private Map<String, Integer> var2NumParamAddedNoise = new HashMap<>();
    // stores the index of PROPER priors
    private Map<String, Integer> properPriorDims = new HashMap<>();

    // stores the name of the distributions whose prior has been assigned (except by @limit)
    private HashSet<String> assigned = new HashSet<String>();

    // stores variables names that bind to distributions whose support depend on its parameter i.e. noise
    private Map<String, String> varsWithFlexSupport = new HashMap<>();

    private Map<String, ArrayList<Pair<Number, Number>>> noiseBound = new HashMap<>();

    private StringBuilder rvInfoCode = new StringBuilder();

    private StringBuilder boundsCode = new StringBuilder();
    private StringBuilder senseCode = new StringBuilder();

    // sense ends -----------------------------------------------------------------------------------------------
    private StringBuilder dataSection = new StringBuilder("    def init_data(self):\n        pass\n");
    private StringBuilder torchCode = new StringBuilder();
    private StringBuilder torchBeginCode = new StringBuilder("    def analysis(self,adaptive_bounds, noise):\n        splits = self.splits\n");
    private StringBuilder endCode = new StringBuilder();
    private Map<String, Integer> paramDims = new HashMap<>();
    private String[] paramDims_keySet = null;
    private Map<String, Integer> dataDims = new HashMap<>();
    private PytorchVisitor pytorchVisitor = new PytorchVisitor();
    private int[] shape;
    private String densityCube_p_branch = "densityCube_p";
    private Map<BasicBlock, String> blockDensityCubeMap = new HashMap<>();
    // true_block -> densityCube_p_true,
    // false_block -> densityCube_p_false,
    // join_block -> densityCube_p = log(exp(densityCube_p_true) + exp(densityCube_p_false))
    private HashMap<BasicBlock, String> joinBlock = new HashMap<>();
    // join_block -> theta
    private Stack<String> ifCondStack = new Stack<>();
    // latest if cond theta
    private Map<String, Pair<String, String>> bounds = new HashMap<>();
    // Cases:
    // 1. given bounds or inferred: (lower, upper)
    // 2. not given bounds nor prior: no such key in bounds, but key in paramDims
    // 3. not given bounds but prior: adaptive_lower, adaptive_upper, key in adaptive_bounds
    // 4. discrete given: ([0, 1], null)
    // 5. dependent var: null in bounds, but not in paramDims
    private HashMap<BasicBlock, String> loopBody = new HashMap<>();
    private HashMap<String, Integer> paramVecLength = new HashMap<>();
    private StringBuilder findboundCode = new StringBuilder("    def find_bounds(self):\n");
    private StringBuilder beginCode = new StringBuilder("import torch\n" +
            "from torch import multiprocessing as mp\n" +
            "import sys\n" +
            "import os\n" +
            "import argparse\n" +
            "import torch.distributions as tdist\n" +
            "import matplotlib.pyplot as plt\n" +
            "from numpy import savetxt\n" +
            "from numpy import matrix\n" +
            "from sympy.plotting.plot import MatplotlibBackend, Plot\n" +
            "from sympy.plotting import plot \n" +
            "from sympy import Symbol\n" +
            "from functools import reduce\n" +
            "sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../../')\n" +
            "import metrics\n" +
            "\n" +
            "torch.set_printoptions(precision=8, sci_mode=False)\n" +
            "torch.distributions.Distribution.set_default_validate_args(False)\n" +
            "\n" +
            "\n" +
            "class Analysis:\n" +
            "    def __init__(self, noise_bound, num_params, splits = 50, num_noise = 100, is_cpu=False, optimize = False):\n" +
            "        if is_cpu:\n" +
            "            self.device = torch.device(\"cpu\")\n" +
            "        else:\n" +
            "            self.device = torch.device(\"cuda:0\")\n" +
            "        self.splits = splits\n" +
            "        self.noise_bound = noise_bound\n" +
            "        self.num_noise = num_noise\n" +
            "        self.optimize = optimize\n" +
            "        self.zero_noise = torch.zeros(num_params, device=self.device)\n" +
            "        self.init_data()\n" +
            "        self.bounds = self.find_bounds()\n\n"
            );
    private Map<String, Pair<String, String>> adaptive_bounds = new HashMap<>();
    private HashMap<String, String> sigmaSet = new HashMap<>(); // sigma -> 0, for param with bounds (0, inf) but needs adaption
    private String indent = "        ";
    private Map<String, Number> constData = new HashMap<>();
    private HashMap<String, Integer> robustParams = new HashMap<>();
    private boolean inLoop = false;

    public String runCompiler(String tempfilePath) {
        aqua.cfg.CFGBuilder cfgBuilder;
        if (!getCFGPlot)
            cfgBuilder = new CFGBuilder(tempfilePath, null, false); // null,false "target/tempfile.png"
        else
            cfgBuilder = new CFGBuilder(tempfilePath, "target/tempfile.png", true); // null,false
        ArrayList<Section> CFG = cfgBuilder.getSections();
        pytorchVisitor.dataDims = dataDims;
        pytorchVisitor.robustParams = robustParams;
        // if (!addEps)
        torchCode.append(indent).append("ma_eps = 1.0E-9\n");
        torchCode.append(indent).append("densityCube_p = torch.zeros(1, device=self.device)\n");
        // else
        //     torchCode.append(indent).append("    ").append("densityCube_p = torch.zeros(1, device=self.device)\n");
        // Outer loop
        if (getDataGrad) {
            beginCode.append(
                    "        self.query = sys.argv[3]\n" +
                            "\n");
        }
        if (!getDataGrad) {
//            endCode.append("    @torch.no_grad()\n");
        }
        endCode.append("    def run_analysis(self, noise):\n");

        if (enableTiming) {
            endCode.append("        import time\n");
            endCode.append(
                    "        # the program can run faster if removing all timing code\n" +
                            "        start = torch.cuda.Event(enable_timing=True)\n" +
                            "        end = torch.cuda.Event(enable_timing=True)\n" +
                            "        start.record()\n");
        }
//        endCode.append("        self.init_data()\n");
        if (getDataGrad) {
            endCode.append("        self.delta = torch.zeros_like(self.y, device=self.device, requires_grad=True)\n");

        }
        if (!getDataGrad) {
//            endCode.append("        bounds = self.find_bounds()\n");
        } else {

            endCode.append("        with torch.no_grad():\n");
            endCode.append("            bounds = self.find_bounds()\n");
        }

        // Main Body
        getTorchCode(CFG);

        torchCode.append("\n");
        torchCode.append(indent).append("expDensityCube_p = torch.exp(densityCube_p - torch.max(densityCube_p).item())\n").append(indent).append("z_expDensityCube = torch.sum(expDensityCube_p)\n");
        int[] dimSet = new int[shape.length - 1];
        for (int i = 0; i < dimSet.length; i++) {
            dimSet[i] = i + 1;
        }
        for (Map.Entry<String, Integer> paramIdDim : paramDims.entrySet()) {
            Integer currDim = paramIdDim.getValue();
            int toReplace = currDim - 1;
            int orgDim = dimSet[toReplace];
            dimSet[toReplace] = 0;
            torchCode.append(indent).append(
                    String.format("posterior_%1$s = expDensityCube_p.sum(%2$s) / z_expDensityCube\n",
                            paramIdDim.getKey(), Arrays.toString(dimSet))
            );
            dimSet[toReplace] = orgDim;
        }

        // set adaptive bounds
        indent = "        ";
        torchCode.append(indent).append("_ret = {}\n");

        StringBuilder boundsToReturn = new StringBuilder();
        StringBuilder paramsToGet = new StringBuilder();
        StringBuilder paramsToReturn = new StringBuilder();
        Set<String> paramDims_keySet_temp = paramDims.keySet();

        ArrayList<String> sortedVars = new ArrayList<>();
        paramDims.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .forEachOrdered(x -> sortedVars.add(x.getKey()));

        // sense, sort the vars based on their index, so that analysis returns (x, px)* in order
        paramDims_keySet = paramDims_keySet_temp.toArray(new String[paramDims_keySet_temp.size()]);
        if (adaptive_bounds.size() > 0) {
            findboundCode.append(indent).append("repeat = True\n");
        } else {
            findboundCode.append(indent).append("repeat = False\n");
        }
        for (String paramId : sortedVars) {
            if (adaptive_bounds.containsKey(paramId)) {
                Pair<String, String> init_lower_upper = adaptive_bounds.get(paramId);
                zeroLowerToEps(init_lower_upper);
                findboundCode.append(indent).append(String.format("adaptiveLower_%1$s = %2$s\n", paramId, init_lower_upper.getKey()));
                findboundCode.append(indent).append(String.format("adaptiveUpper_%1$s = %2$s\n", paramId, init_lower_upper.getValue()));
                boundsToReturn.append(String.format("adaptiveUpper_%1$s,adaptiveLower_%1$s,", paramId));
            }
            paramsToGet.append(String.format("%1$s,posterior_%1$s,", paramId));
            torchCode.append(indent).append(String.format("_ret[\"%1$s\"] = (%1$s.flatten(),posterior_%1$s.flatten())\n", paramId));
        }
        findboundCode.append(indent).append("while repeat:\n");


        // add eps
        // if (addEps) {
        //     dataSection.append("\neps = float(sys.argv[2])\n");
        //     dataSection.append("\nfor eps_i in range(len(y)):\n" +
        //             "    y[eps_i] += eps\n\n");
        // }

        String moreIndent = "";
        // if (addEps) {
        //     moreIndent = "    ";
        // }
        // String innerSplits = "50";
        // if (sameInnerSplits) {
        //     innerSplits = "int(sys.argv[1])";
        // }
        // dataSection.append("\n" + moreIndent + "while True:\n" +
        //         moreIndent + "    if repeat:\n" +
        //         moreIndent + "        splits = " + innerSplits + "\n" +
        //         moreIndent + "    else:\n" +
        //         moreIndent + "        splits = int(sys.argv[1])\n");

        // for (String paramId : paramDims_keySet) {
        //     torchCode.append(indent).append(String.format("%1$s = %1$s.flatten()\n", paramId));
        //     torchCode.append(indent).append(String.format("posterior_%1$s = posterior_%1$s.flatten()\n", paramId));
        // }
        // findboundCode.append(indent).append("if repeat == False:\n").append(indent).append("    break\n\n");


        indent += "    ";
        findboundCode.append(indent).append("repeat = False\n");
        String paramsToGetStr = paramsToGet.substring(0, paramsToGet.length() - 1);

        findboundCode.append(indent).append("ret = self.analysis([").append(boundsToReturn).append("],self.zero_noise)\n");
        for (String paramId : sortedVars) {
            findboundCode.append(indent).append(String.format("%1$s, posterior_%1$s = ret[\"%1$s\"]\n", paramId));
        }

        for (String paramId : paramDims_keySet) {
            if (adaptive_bounds.containsKey(paramId)) {
                findboundCode.append(indent).append(String.format("lowProb_%1$s = posterior_%1$s.max() * 0.001\n", paramId));
                findboundCode.append(indent).append(String.format("all_gt_%1$s = (posterior_%1$s > lowProb_%1$s).nonzero(as_tuple=True)[0]\n", paramId));
                findboundCode.append(indent).append(String.format("if abs(all_gt_%1$s[0] - all_gt_%1$s[-1]) < 2 ", paramId));
                findboundCode.append(String.format("and not ((%1$s[max(all_gt_%1$s[0] - 1, 0)] == adaptiveLower_%1$s)", paramId));
                findboundCode.append(String.format(" or (%1$s[min(all_gt_%1$s[-1] + 1, len(%1$s) - 1)] == adaptiveUpper_%1$s)):\n", paramId));
                findboundCode.append(indent).append("    repeat = True\n");
                findboundCode.append(indent).append(String.format("adaptiveLower_%1$s = %1$s[max(all_gt_%1$s[0] - 1, 0)]\n", paramId));
                findboundCode.append(indent).append(String.format("adaptiveUpper_%1$s = %1$s[min(all_gt_%1$s[-1] + 1, len(%1$s) - 1)]\n", paramId));
            }
        }
        indent = indent.substring(0, indent.length() - 4);
        findboundCode.append(indent).append(String.format("return [%1$s]\n\n", boundsToReturn.toString()));
        String boundsToReturnStr = boundsToReturn.toString();
        if (boundsToReturnStr.length() > 0)
            torchBeginCode.append(indent).append(boundsToReturnStr).append(" = adaptive_bounds\n");
        else
            torchBeginCode.append(indent).append("empty_bounds").append(" = adaptive_bounds\n");
        // set data
        for (String dataID : dataDims.keySet()) {
            dataSection.append(indent).append(String.format("self.%1$s = %1$s\n", dataID));
            torchBeginCode.append(indent).append(String.format("%1$s = self.%1$s\n", dataID));
        }
        if (getDataGrad) {
            torchBeginCode.append(indent).append("y = self.y + self.delta\n");
        }
        for (String dataID : constData.keySet()) {
            dataSection.append(indent).append(String.format("self.%1$s = %1$s\n", dataID));
            torchBeginCode.append(indent).append(String.format("%1$s = self.%1$s\n", dataID));
        }
        dataSection.append("\n");


        torchCode.append(indent).append(String.format("return _ret\n\n"));
        endCode.append(indent).append("torch.cuda.empty_cache()\n");
        endCode.append(indent).append("return self.analysis(self.bounds, noise)\n");


        if (enableTiming)
            endCode.append(indent).append("end.record()\n");
        if (getDataGrad) {
            for (String paramId : paramDims_keySet) {
                endCode.append(String.format("%1$sif self.query == \"%2$s\":\n", indent, paramId));
                indent += "    ";
                endCode.append(String.format("%2$sprint(\"%1$s\", (%1$s * posterior_%1$s).sum().item())\n", paramId, indent));
                // for (String dataId: dataDims.keySet()) {
                //     //if (!constData.containsKey(dataId))
                //     endCode.append(String.format("%1$sself.%2$s.grad = None\n", indent, dataId));
                // }
                endCode.append(String.format("%1$sself.%2$s.grad = None\n", indent, "delta"));
                endCode.append(String.format("%2$sexpectation_%1$s = (%1$s * posterior_%1$s).sum()\n", paramId, indent));
                endCode.append(String.format("%2$sexpectation_%1$s.backward()\n", paramId, indent)); // retain_graph=True only need for querying two param grad
                // for (String dataId: dataDims.keySet()) {
                //     //if (!constData.containsKey(dataId))
                //     endCode.append(String.format("%2$sprint(\"dE[%1$s]/d%3$s\", self.%3$s.grad.flatten())\n", paramId, indent, dataId));
                // }
//                endCode.append(String.format("%2$sprint(\"dE[%1$s]/dy\", self.delta.grad.flatten())\n", paramId, indent));
            indent = indent.substring(0, indent.length() - 4);
            }
        } else {
//            for (String paramId : paramDims_keySet) {
//                endCode.append(String.format("%2$sprint(\"E[%1$s]\", (%1$s * posterior_%1$s).sum().item())\n", paramId, indent));
//            }
        }
        // for (String paramId :paramDims.keySet()) {
        //     endCode.append(String.format("print(%1$s.flatten())\n", paramId));
        //     endCode.append(String.format("print(posterior_%1$s)\n", paramId));
        // }
//        endCode.append(indent).append("MS_TO_S = 1/1000\n");

        if (enableTiming) {
            endCode.append(indent).append(
                    "print(\"Time: {:.6f} (s)\".format(start.elapsed_time(end) * MS_TO_S, file=sys.stderr))");
                    // "print(start.elapsed_time(end) * MS_TO_S, file=sys.stderr)\n");
        }

//        endCode.append("\n" +
//                "ana = Analysis() \n" +
//                "ana.run_analysis()\n");


        // sense -----------------------------------------------------
        rvInfoCode.append("\n\n# vars dimension -----------------------------\n").append("vars2Index = {}\n");

        int i = 0;
        for (String s : var2NoiseIndices.keySet()){
            rvInfoCode.append(String.format("vars2Index[\"%1$s\"] = %2$s\n", s, i));
            ++i;
        }
        rvInfoCode.append(String.format("num_param = %1$s\n", paramIndex));

        rvInfoCode.append("vars2NumParams = {}\n");
        for (String s : var2NumParamAddedNoise.keySet()){
            rvInfoCode.append(String.format("vars2NumParams[\"%1$s\"] = %2$s\n", s, var2NumParamAddedNoise.get(s)));
        }
        rvInfoCode.append("# vars dimension ends -----------------------------\n\n");

        boundsCode.append("# noiseBounds ------------------------\n# CHANGE noise_bound if you'd like \nnoise_bound = {}\n");
        for (String s : noiseBound.keySet()){
            boundsCode.append(String.format("noise_bound[\"%1$s\"] = []\n", s));
            for (Pair<Number, Number> p : noiseBound.get(s)){
                boundsCode.append(String.format("noise_bound[\"%1$s\"].append([%2$s,%3$s])\n", s, p.getKey(), p.getValue()));
            }
        }
        boundsCode.append("# noiseBounds ends -----------------\n");

        senseCode = new StringBuilder("\n\n" +
                "# https://stackoverflow.com/questions/53289609/how-to-graph-points-with-sympy\n" +
                "def get_sympy_subplots(plot: Plot):\n" +
                "    backend = MatplotlibBackend(plot)\n" +
                "\n" +
                "    backend.process_series()\n" +
                "    backend.fig.tight_layout()\n" +
                "    return backend.fig, backend.ax[0]\n\n" +
                "# return a noise tensor of the shape M x N, M = num_noise^num_params, N = num_params\n" +
                "# each row is a noise vector to analysis\n" +
                "def noise_matrix(isjoint, var, param, param_index, num_noise, iscpu):\n" +
                "    if not iscpu:\n" +
                "        device = torch.device(\"cuda:0\")\n" +
                "    else:\n" +
                "        device = torch.device(\"cpu\")\n" +
                "\n" +
                "    if (isjoint):\n" +
                "        lst_tensors = []\n" +
                "        for v in vars2Index.keys():\n" +
                "            for bound in noise_bound[v]:\n" +
                "                lst_tensors.append(torch.arange(bound[0], bound[1], step=(bound[1] - bound[0])/num_noise, device = device))\n" +
                "        return torch.cartesian_prod(*lst_tensors)\n" +
                "    else:\n" +
                "        m = torch.zeros(num_noise * num_param, device = device)\n" +
                "        m = torch.reshape(m, [num_noise, num_param])\n" +
                "\n" +
                "        my_noise = torch.arange(noise_bound[var][param][0], noise_bound[var][param][1], step=(noise_bound[var][param][1] - noise_bound[var][param][0])/num_noise, device = device)\n" +
                "\n" +
                "        for i in range(num_noise):\n" +
                "            m[i][param_index] = my_noise[i]\n" +
                "\n" +
                "        return m\n" +
                "\n" +
                "def sensitivity_wrapper(done_queue, done, *args):\n" +
                "    ret = sensitivity(*args)\n" +
                "    done_queue.put(ret)\n" +
                "    done.wait()\n" +
                "\n" +
                "def sensitivity(nm, dist, is_data, is_cpu, randvar, param, param_index, is_joint, split, num_noise):\n" +
                "\n" +
                "    ana = Analysis(noise_bound, num_param, split, num_noise, is_cpu)\n" +
                "    \n" +
                "    ret = ana.run_analysis(torch.zeros(num_param, device=ana.device))\n" +
                "    x, px = ret[randvar]\n" +
                "    base_exp = (x * px).sum().item()\n" +
                "\n" +
                "    #print(\"Expectation: \", base_exp)\n" +
                "    #print(\"-----------------\")\n" +
                "    #print(\"Support \\n\", x)\n" +
                "    #print(\"-----------------\")\n" +
                "    #print(\"Posterior: \\n\", px)\n" +
                "\n" +
                "    distances = []\n" +
                "\n" +
                "    for i in range(num_noise):\n" +
                "        n = nm[i, :]\n" +
                "        ret = ana.run_analysis(n)\n" +
                "\n" +
                "        x_noise, px_noise = ret[randvar]\n" +
                "\n" +
                "        distances.append(metrics.metr[dist](torch.clone(x), torch.clone(px), x_noise, px_noise))\n" +
                "\n" +
                "    # store to .csv file\n" +
                "    noises = list(nm[:,param_index].tolist())\n" +
                "    csv = os.path.dirname(os.path.realpath(__file__)) + \"/sense_\" + __file__.split(\"/\")[-1][:-3] +\"_\"+ randvar + \"_\" + (str)(param+1) + \"_\" + dist + \".csv\"\n" +
                "    savetxt(csv, matrix([noises, distances]), delimiter=',')\n" +
                "    print(\"Saved to \", csv)\n" +
                "\n" +
                "if __name__ == \"__main__\":\n" +
                "    parser = argparse.ArgumentParser(prog=\"Sense\", description='Sensitivity analysis tool for probabilistic programs in Stan/Storm, based on quantized inference')\n" +
                "    \n" +
                "    # measure distance OR optimize distance\n" +
                "    # if no arg, use measure\n" +
                "    parser.add_argument(\"-o\", \"--optimize\", action=\"store_true\", help=\"Use this option to maximize distance w.r.t. noise, using the default ADAM optimizer\")\n" +
                "\n" +
                "    # specify if use pytorch multiprocessing for all vars and params in the model\n" +
                "    # default is false\n" +
                "    parser.add_argument(\"-a\", \"--all\", action=\"store_true\", help=\"Use this option to run aquaSense on all params in the model at once\")" +
                "\n" +
                "    # specify distance metric\n" +
                "    # of no arg, use ED1\n" +
                "    parser.add_argument(\"-d\", \"--metric\", nargs=1, choices=[\"expdist1\", \"expdist2\", \"KS\"], help=\"Specify the distance metric to be used, default is ED1\")\n" +
                "    \n" +
                "    # specify whether analysis is w.r.t noise in prior OR data\n" +
                "    # if no arg, prior\n" +
                "    parser.add_argument(\"-data\", action=\"store_true\", help=\"Use this option to specify noise in data instead of prior\")\n" +
                "\n" +
                "    # specify whether to use CPU\n" +
                "    # if no arg, use GPU\n" +
                "    parser.add_argument(\"-c\", \"--cpu\", action=\"store_true\", help=\"Use this option use CPU instead of GPU\")\n" +
                "\n" +
                "    # specify which variable to add noise to\n" +
                "    # if no arg, use the first randvar\n" +
                "    parser.add_argument(\"-v\", \"--randvar\", choices=list(vars2Index.keys()), help=\"Use this option to specify the random variable to add noise\")\n" +
                "\n" +
                "    # specify which parameter to add noise to\n" +
                "    # if no arg, use the first parameter\n" +
                "    parser.add_argument(\"-p\", \"--parameter\", nargs=1, type=int, help=\"Use this option to specify the parameter index to add noise\")\n" +
                "    \n" +
                "    # specify if use the joint distribution over all prior params\n" +
                "    parser.add_argument(\"-j\", \"--joint\", action=\"store_true\", help=\"Use this option to analyze the joint posterior with all prior parameters with noise, suppreses -p\")\n" +
                "    \n" +
                "    # specify the #splits\n" +
                "    parser.add_argument(\"-s\", \"--split\", type=int, default=50, help=\"Use this option to specify the granularity of quantized inference, default is 50\")\n" +
                "    \n" +
                "    parser.add_argument(\"-n\", \"--noise\", type=int, default=50, help=\"Use this option to specify the # points of noise, default is 50\")\n" +
                "    \n" +
                "    parser.add_argument(\"-b\", \"--bound\", nargs=2, type=float, help=\"Use this option to specify the lower & upper bound of eps, e.g. -v beta -p 0 -b -0.5 0.5\")\n" +
                "    args = parser.parse_args()\n" +
                "\n" +
                "    randvar = None\n" +
                "    for k in list(vars2Index.keys()):\n" +
                "        if (vars2Index[k] == 0):\n" +
                "            randvar = k\n" +
                "    parameter = 0\n" +
                "    metric = \"expdist1\"\n" +
                "\n" +
                "    if (args.randvar != None):\n" +
                "        randvar = args.randvar\n" +
                "    \n" +
                "    if (args.parameter != None):\n" +
                "        parameter = args.parameter[0]\n" +
                "\n" +
                "    if (args.bound != None):\n" +
                "        if (args.bound[0] >= args.bound[1]):\n" +
                "            sys.exit(\"Error: invalid noise bound\")\n" +
                "        noise_bound[randvar][parameter] = args.bound\n" +
                "\n" +
                "    if (args.metric != None):\n" +
                "        metric = args.metric[0]\n" +
                "\n" +
                "    # check for validity of cmd arguments\n" +
                "    if (parameter < 0 or parameter > len(noise_bound[randvar])):\n" +
                "        sys.exit(\"Error: illegal parameter index\")\n" +
                "    \n" +
                "    if not args.cpu:\n" +
                "        start = torch.cuda.Event(enable_timing=True)\n" +
                "        endall = torch.cuda.Event(enable_timing=True)\n" +
                "        start.record()\n" +
                "        torch.zeros(1, device=torch.device(\"cuda:0\"))\n" +
                "        endall.record()\n" +
                "        torch.cuda.synchronize()\n" +
                "        print(\"Setting up GPU, Torch Took: {:.6f}\".format(start.elapsed_time(endall) /1000, file=sys.stderr))\n" +
                "    \n" +
                "    # compute the global param index\n" +
                "    param_index = reduce(lambda x, y: x + y, map(lambda k: len(noise_bound[k]), filter(lambda var: vars2Index[var] < vars2Index[randvar], vars2Index.keys())), 0) + parameter\n" +
                "    \n" +
                "    nm = noise_matrix(args.joint, randvar, parameter, param_index, args.noise, args.cpu)\n" +
                "    \n" +
                "    \n" +
                "    if (args.optimize):\n" +
                "        pass\n" +
                "    elif (args.all):\n" +
                "        for randvar in vars2Index:\n" +
                "            for parameter in range(len(noise_bound[randvar])):\n" +
                "                index = reduce(lambda x, y: x + y, map(lambda k: len(noise_bound[k]), filter(lambda var: vars2Index[var] < vars2Index[randvar], vars2Index.keys())), 0) + parameter\n" +
                "                nm = noise_matrix(args.joint, randvar, parameter, index, args.noise, args.cpu)\n" +
                "                sensitivity(nm, metric, args.data, args.cpu, randvar, parameter, param_index, args.joint, args.split, args.noise)\n" +
                "    else:\n" +
                "        nm = noise_matrix(args.joint, randvar, parameter, param_index, args.noise, args.cpu)    \n" +
                "        # def sensitivity(dist, is_data, is_cpu, randvar, param, is_joint, split, num_noise):\n" +
                "        sensitivity(nm, metric, args.data, args.cpu, randvar, parameter, param_index, args.joint, args.split, args.noise)");


        endCode.append(rvInfoCode).append(boundsCode).append(senseCode);

        // -----------------------------------------------------------

        // Get python code
        // System.out.println(beginCode.append(dataSection).append(torchCode).append(endCode));
        // System.out.println(torchCode);
        // System.out.println(endCode);
        beginCode.append(dataSection).append(torchBeginCode).append(torchCode).append(findboundCode).append(endCode);
        return beginCode.toString();

        // for (Map.Entry<String, Pair<String, String>> pair:adaptive_bounds.entrySet()) {
        //     System.out.println(String.format("%s:(%s,%s)", pair.getKey(), pair.getValue().getKey(), pair.getValue().getValue()));
        // }

    }


    public void getTorchCode(ArrayList<Section> CFG) {
        // get dims for each var
        int dimCounter = 0;
        for(Section section: CFG) {
            if (section.sectionType == SectionType.DATA) {
                ArrayList<AST.Data> dataSets = section.basicBlocks.get(0).getData();
                for (AST.Data data: dataSets) {
                    if (data.vector != null || data.array != null) {
                        dataDims.put(data.decl.id.id, dimCounter);
                        //dimCounter += 1;
                    } else {
                        // TODO const float data
                        if (data.expression instanceof AST.Integer ) {
                            constData.put(data.decl.id.id, Integer.valueOf(data.expression.toString()));
                        } else if (data.expression instanceof AST.Double) {
                            constData.put(data.decl.id.id, Double.valueOf(data.expression.toString()));

                        }
                    }
                }
                // TODO: two data arrays
            }
            else if (section.sectionType == SectionType.FUNCTION) {
                if (section.sectionName.equals("main")) {
                    dimCounter += 1;
                    for (BasicBlock basicBlock: section.basicBlocks) {
                        for (Statement statement : basicBlock.getStatements()) {
                            if (statement.statement instanceof AST.Decl) {
                                dimCounter = processDecl(dimCounter, statement);
                            }
                            else if (statement.statement instanceof AST.AssignmentStatement) {
                                AST.AssignmentStatement assignmentStatement = (AST.AssignmentStatement) statement.statement;
                                pytorchVisitor.update_evaluate(var2NoiseIndices, noiseBound, var2NumParamAddedNoise);
                                String lhs = pytorchVisitor.evaluate(assignmentStatement.lhs);

                                // sense -----------------------------------------
                                // if lhs is a singleton variable
                                // e.g.
                                // float a
                                // a = normal(0,1)
                                if (paramDims.containsKey(lhs)) {

                                    String funcId = null;
                                    if (assignmentStatement.rhs instanceof AST.FunctionCall) {
                                        AST.FunctionCall functionCall = (AST.FunctionCall) assignmentStatement.rhs;
                                        funcId = functionCall.id.id;
                                        if (!functionCall.isDistribution && !otherDists.contains(funcId)) {
                                            // x = bernoulli(~) + bernoulli(~)
                                            // rhs is not a single distribution, not supported for now
                                            funcId = null;
                                        } else {
                                            getBoundsFromPrior(lhs, functionCall);
                                        }
                                    }
                                    if (funcId == null && !bounds.containsKey(lhs)) { // dependent var, if rhs is not distr or functionCall
                                        paramDims.remove(lhs);
                                        bounds.put(lhs, null);
                                    }
                                }
                                // sense --------------------------------------------
                                // if lhs is an indexed parameter, e.g.
                                // parameter{
                                //     int<lower=0, upper=1> mu[
                                // }
                                // model {
                                //  for (d in 1:D) {
                                //      mu[d] = normal(0,100)
                                //  }
                                //  ...
                                // }
                                //
                                // OR
                                //
                                // if lhs is a vectorized variable, record noise bounds for each entry independently
                                // e.g.
                                // float a[2]
                                // a = normal(0,1)
                                else if (paramDims.containsKey(String.format("%1$s[0]", lhs)) ||
                                        (assignmentStatement.lhs instanceof AST.ArrayAccess &&
                                                paramDims.containsKey(((AST.ArrayAccess)assignmentStatement.lhs).id + "[0]"))){

                                    int dim = 0;

                                    // sanitize lhs, get rid of indexing
                                    if (lhs.contains("[")){
                                        lhs = lhs.substring(0, lhs.indexOf('['));
                                    }

                                    while (paramDims.containsKey(String.format("%1$s[%2$s]", lhs, dim))){
                                        String lhsWithIndex = String.format("%1$s[%2$s]", lhs, dim);
                                        String funcId = null;
                                        if (assignmentStatement.rhs instanceof AST.FunctionCall) {
                                            AST.FunctionCall functionCall = (AST.FunctionCall) assignmentStatement.rhs;
                                            funcId = functionCall.id.id;
                                            if (!functionCall.isDistribution && !otherDists.contains(funcId)) {
                                                funcId = null;
                                            } else {
                                                getBoundsFromPrior(lhsWithIndex, functionCall);
                                            }
                                        }
                                        if (funcId == null && !bounds.containsKey(lhsWithIndex)) { // dependent var, if rhs is not distr or functionCall
                                            paramDims.remove(lhsWithIndex);
                                            bounds.put(lhsWithIndex, null);
                                        }
                                        dim++;
                                    }
                                }
                                // if lhs is Data i.e. this is an observe statement
                                // e.g.
                                //
                                //
                                else {
                                    int i = 0;
                                    String funcId = null;
                                    if (assignmentStatement.rhs instanceof AST.FunctionCall) {
                                        AST.FunctionCall functionCall = (AST.FunctionCall) assignmentStatement.rhs;
                                        funcId = functionCall.id.id;
                                        if (!functionCall.isDistribution && !otherDists.contains(funcId)) {
                                            funcId = null;
                                        } else {
                                            // observe statement DO NOT NEED prior/bound
                                            // getBoundsFromPrior(lhs, functionCall);
                                        }
                                    }
                                    if (funcId == null && !bounds.containsKey(lhs)) { // dependent var, if rhs is not distr or functionCall
                                        paramDims.remove(lhs);
                                        bounds.put(lhs, null);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Cannot query for an array element in Storm, e.g. posterior(beta[1])
            // so don't use this
            // else if (section.sectionType == SectionType.QUERIES){
            //     ArrayList<AST.Query> queries = section.basicBlocks.get(0).getQueries();
            //     for (AST.Query query: queries) {
            //         System.out.println(query.toString());
            //     }
            // }
        }
        // init shape array
        shape = new int[dimCounter];
        for (int i = 0; i < dimCounter; i++)
            shape[i] = 1;
        //sections from CFG Builder
        for(Section section: CFG) {
            if (section.sectionType == SectionType.DATA) {
                indent = "        ";
                getDataString(section.basicBlocks.get(0).getData());

            }
            else if (section.sectionType == SectionType.FUNCTION) {
                if (section.sectionName.equals("main")) {
                    // if (!addEps)
                    indent = "        ";
                    // else
                    //     indent = "        ";
                    this.visitBlocks(section.basicBlocks);
                }
            }
        }
    }

    // sense
    private void properPriorAddNoiseBound(String id, Pair<String, String> upper_lower){

        String lo = upper_lower.getKey();
        String up = upper_lower.getValue();
        Double noiseRatioLo = -0.1;
        Double noiseRatioHg = 0.1;

        Pair<Number, Number> Bound1 = new Pair<>(-3, 3);
        Pair<Number, Number> Bound2 = new Pair<>(-3, 3);

        try {
            double p1 = Double.parseDouble(lo);
            double p2 = Double.parseDouble(up);
            if (p1 != 0 && p2 != 0){
                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
            } else if (p1 != 0){
                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                Bound2 = Bound1;
            } else {
                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                Bound1 = Bound2;
            }
        } catch (NumberFormatException ignored) {}

        noiseBound.put(id, new ArrayList<>(Arrays.asList(Bound1, Bound2)));
    }

    private int processDecl(int dimCounter, Statement statement) {
        AST.Decl decl = (AST.Decl) statement.statement;
        if (decl.dtype.primitive == AST.Primitive.VECTOR ||
                (decl.dtype.primitive == AST.Primitive.FLOAT && decl.dims != null)){
            if (decl.id.id.startsWith("robust_")) {
                Pair<String, String> lower_upper = getLimits(decl);
                assert lower_upper != null;
                // if (lower_upper.getKey().equals("0"))
                //     lower_upper.setKey("ma_eps");
                robustParams.put(decl.id.id, dimCounter);
                dimCounter += 1;
                bounds.put(decl.id.id, lower_upper);
            }
            else {
                Integer dim = getDim(decl);
                Pair<String, String> lower_upper = getLimits(decl);
                for (int i = 0; i < dim; i++) {
                    String id = String.format("%1$s[%2$s]", decl.id.id, i);
                    paramDims.put(id, dimCounter);
                    if (lower_upper != null) {

                        // sense -------------------------------------
                        // proper prior
                        varsWithFlexSupport.put(id, "uniform");

                        // update param2NoiseIndex map
                        var2NoiseIndices.put(id, IntStream.range(paramIndex, paramIndex + 2)
                                .boxed()
                                .collect(Collectors.toList()));

                        // update priorPriorDims
                        properPriorDims.put(id, var2NoiseIndices.size() - 1);

                        paramIndex += 2;

                        // add proper prior to noise_bound
                        properPriorAddNoiseBound(id, lower_upper);
                        // sense ---------------------------------------

                        if (!adaptive)
                            bounds.put(id, lower_upper);
                        else
                            adaptive_bounds.put(id, lower_upper);
                    }
                    // else infer from priors and adapt
                    dimCounter += 1;
                }
            }
        }
        else {
            Pair<String, String> lower_upper = getLimits(decl);
            String id = decl.id.id;
            if (lower_upper != null) {
                // sense ----------------------------------------

                // proper prior
                varsWithFlexSupport.put(id, "uniform");

                // update param2NoiseIndex map
                var2NoiseIndices.put(id, IntStream.range(paramIndex, paramIndex + 2)
                        .boxed()
                        .collect(Collectors.toList()));

                // update priorPriorDims
                properPriorDims.put(id, var2NoiseIndices.size() - 1);

                paramIndex += 2;

                // add proper prior to noise_bound
                properPriorAddNoiseBound(id, lower_upper);
                // sense ----------------------------------------

                if (!adaptive)
                    bounds.put(decl.id.id, lower_upper);
                else
                    adaptive_bounds.put(decl.id.id, lower_upper);
            }
            // else infer from priors and adapt
            paramDims.put(decl.id.id, dimCounter);
            dimCounter += 1;
        }
        return dimCounter;
    }

    private Integer getDim(AST.Decl decl) {
        Integer dim;
        if (decl.dtype.primitive == AST.Primitive.FLOAT) {
            String dimInt = decl.dims.dims.get(0).toString();
            if (constData.containsKey(dimInt))
                dim = (int) constData.get(dimInt);
            else
                dim = Integer.valueOf(dimInt);
        }
        else {
            String dimInt = decl.dtype.dims.dims.get(0).toString();
            if (constData.containsKey(dimInt))
                dim = (int) constData.get(dimInt);
            else
                dim = Integer.valueOf(dimInt);
        }
        return dim;
    }

    private void getBoundsFromPrior(String lhs, AST.FunctionCall functionCall) {
        // bounds not given, decide intervals based on distributions
        if (!bounds.containsKey(lhs)) {
            if (!adaptive) {
                Pair<String, String> rhs_bounds = getBoundedDistrBounds(lhs, functionCall);
                if (rhs_bounds != null) // put the infered bounds, e.g. uniform(0, 10)
                    bounds.put(lhs, rhs_bounds);
                else { // if normal(0, 1) use adaptive bounds
                    rhs_bounds = new Pair<>(
                            String.format("adaptiveLower_%1$s", lhs),
                            String.format("adaptiveUpper_%1$s", lhs));
                    bounds.put(lhs, rhs_bounds);
                    Pair<String, String> init_bounds = getUnboundedDistrInitBounds(lhs, functionCall);
                    String rawId = lhs.split("\\[", 2)[0];
                    if (sigmaSet.containsKey(rawId))
                        init_bounds.setKey(sigmaSet.get(rawId));
                    adaptive_bounds.put(lhs, init_bounds);
                }
            } else {
                Pair<String, String> rhs_bounds = new Pair<>(
                        String.format("adaptiveLower_%1$s", lhs),
                        String.format("adaptiveUpper_%1$s", lhs));
                bounds.put(lhs, rhs_bounds);
                // adaptive_bounds (init_bounds) is added in processDecl()
            }
        }
        // if the probabilistic program supplies bounds
        // e.g.
        // @limits<lower=-10, upper=10>
        // float trueX
        //
        // still need to store paramIndex for the sake of noise instrumentation
        else{
            Pair<String, String> rhs_bounds = getBoundedDistrBounds(lhs, functionCall);
            if (rhs_bounds == null) {
                getUnboundedDistrInitBounds(lhs, functionCall);
            }
        }
    }

    private void resetProperPriorIndex(String lhs, int num_params){
        List<Integer> orig_nos_ind = var2NoiseIndices.get(lhs);
        int len = 2;
        if (len == num_params){
            // nothing
        }
        else {
            // initialized to a distribution with more than 2 parameters
            // needs to shift param2NoiseIndex
            int shift = num_params - len;
            var2NoiseIndices.put(lhs, IntStream.range(orig_nos_ind.get(0), orig_nos_ind.get(0) + num_params)
                    .boxed()
                    .collect(Collectors.toList()));
            for (String s : var2NoiseIndices.keySet()){
                if (var2NoiseIndices.get(s).get(0) > orig_nos_ind.get(0)){
                    int ll = var2NoiseIndices.get(s).size();
                    var2NoiseIndices.put(s, IntStream.range(var2NoiseIndices.get(s).get(0) + shift, var2NoiseIndices.get(s).get(0) + shift + ll)
                            .boxed()
                            .collect(Collectors.toList()));
                }
            }
            paramIndex = paramIndex + shift;
        }
    }

    private Pair<String,String> getUnboundedDistrInitBounds(String lhs, AST.FunctionCall functionCall) {
        String funcId = functionCall.id.id;
        Pair<String,String> ret = null;
        int num_params = 0;
        if (funcId.equals("normal") || funcId.equals("gauss")) {
            num_params = 2;

            Double mean = (double) 0;
            Double sd = 1.0;
            try {
                mean = Double.valueOf(functionCall.parameters.get(0).toString());
                sd = Double.valueOf(functionCall.parameters.get(1).toString());
            } catch (NumberFormatException nfe) {

            }
            ret = new Pair<>(String.valueOf(mean - 6 * sd), String.valueOf(mean + 6 * sd));
        }
        else if (funcId.equals("gamma")) {
            num_params = 2;
            ret = new Pair<>("0", "50");
        }
        else if (funcId.equals("geometric")) {
            num_params = 1;
            Double p = (double) 0;
            Double ub = (double) 10;
            try {
                p = Double.valueOf(functionCall.parameters.get(0).toString());
                ub = 6/p;
            } catch (NumberFormatException nfe) {

            }
            ret = new Pair<>("0", String.valueOf(ub));
        }
        else {
            throw new UnsupportedOperationException(funcId + " distribution is not supported!");
        }

        // if number of parameters is not set, report error
        assert(num_params != 0);

        String llhs = lhs;
        if (lhs.contains(".")){
            llhs = lhs.substring(0, lhs.indexOf('.'));
        }

        // case 1: re-assign prior (under the same distribution)
        // float A
        // A = bernoulli(0.5)
        // A = bernoulli(0.6) <--- getBoundsFromPrior()

        // case 2: param is proper (has upper and lower bound)
        // @limits<lower=-10, upper=10>
        // float C
        // C = bernoulli(0.5) <--- getBoundsFromPrior()

        // case 3: param is improper (lacks bound until assign) and not assigned
        // float B
        // B = bernoulli(0.5) <--- getBoundsFromPrior()


        // case 1
        if (assigned.contains(llhs)){
            List<Integer> params = var2NoiseIndices.get(llhs);
            for (int i = paramIndex; i < paramIndex + num_params; i++){
                params.add(i);
            }
//            param2NoiseIndex.put(llhs, params);
            // update priorPriorDims
            properPriorDims.put(llhs, var2NoiseIndices.size() - 1);

            paramIndex += num_params;

        }
        // case 2
        else if (properPriorDims.containsKey(llhs)){
            varsWithFlexSupport.remove(lhs);
            resetProperPriorIndex(lhs, num_params);

            // remove noiseBound of the proper prior
            // e.g. noiseBound["C"] = [[-0.1,0.1], [-0.1,0.1]]
            // needs to pop the first 2 entries out
            noiseBound.get(lhs).remove(0);
            noiseBound.get(lhs).remove(0);
        }
        // cases 3
        else{
            // update param2NoiseIndex map
            var2NoiseIndices.put(llhs, IntStream.range(paramIndex, paramIndex + num_params)
                    .boxed()
                    .collect(Collectors.toList()));

            // update priorPriorDims
            properPriorDims.put(llhs, var2NoiseIndices.size() - 1);

            paramIndex += num_params;

            // mark the var as assigned
            assigned.add(llhs);
        }

        return ret;
    }

    private Pair<String,String> getBoundedDistrBounds(String lhs, AST.FunctionCall functionCall) {
        String funcId = functionCall.id.id;
        Pair<String,String> ret = null;
        boolean isFlex = true;
        int num_params = 0;
        switch (funcId) {
            case "uniform":
            case "uniformClose":
                num_params = 2;
                varsWithFlexSupport.put(lhs, "uniformClose");

                ret = new Pair<>(functionCall.parameters.get(0).toString(), functionCall.parameters.get(1).toString());
                break;
            case "uniformInt": {
                num_params = 2;
                varsWithFlexSupport.put(lhs, "uniformInt");
                int firstInt = Integer.valueOf(functionCall.parameters.get(0).toString());
                int cateLen = 1 - firstInt + Integer.valueOf(functionCall.parameters.get(1).toString());
                int[] categoricalArray = new int[cateLen];
                for (int i = 0; i < cateLen; i++) {
                    categoricalArray[i] = i + firstInt;
                }
                String arrayString = ArrayUtils.toString(categoricalArray);
                arrayString = arrayString.substring(1, arrayString.length() - 1);
                ret = new Pair<>(String.format("[%1$s]", arrayString), null);
                break;
            }
            case "flip":
            case "bernoulli":
            case "bernoulli_logit":
                isFlex = false;
                num_params = 1;
                ret = new Pair<>("[0,1]", null);
                break;
            case "binomial": {
                isFlex = false;
                num_params = 2;
                int cateLen = Integer.valueOf(functionCall.parameters.get(0).toString()) + 1;
                int[] categoricalArray = new int[cateLen];
                for (int i = 0; i < cateLen; i++) {
                    categoricalArray[i] = i;
                }
                String arrayString = ArrayUtils.toString(categoricalArray);
                arrayString = arrayString.substring(1, arrayString.length() - 1);
                ret = new Pair<>(String.format("[%1$s]", arrayString), null);
                break;
            }
            case "atom":
                num_params = 1;
                varsWithFlexSupport.put(lhs, "atom");
                ret = new Pair<>("[" + functionCall.parameters.get(0) + "]", null);
                break;
            case "categorical": {
                isFlex = false;
                num_params = functionCall.parameters.size();
                int cateLen = functionCall.parameters.size();
                int[] categoricalArray = new int[cateLen];
                for (int i = 0; i < cateLen; i++) {
                    categoricalArray[i] = i;
                }
                String arrayString = ArrayUtils.toString(categoricalArray);
                arrayString = arrayString.substring(1, arrayString.length() - 1);
                ret = new Pair<>(String.format("[%1$s]", arrayString), null);
                break;
            }
            case "beta":
                isFlex = false;
                num_params = 2;
                ret = new Pair<>("0", "1");
                break;
            case "discrete":
                num_params = functionCall.parameters.size();
                varsWithFlexSupport.put(lhs, "discrete");
                ret = new Pair<>(functionCall.parameters.toString(), null);
                break;
            case "poisson":
                isFlex = false;
                // note poisson is actually unbounded, its support is (0, +\infty)
                num_params = 2;
                double mean = (double) 5;
                try {
                    mean = Double.parseDouble(functionCall.parameters.get(0).toString());
                } catch (NumberFormatException nfe) {
                }

                int[] arr = new int[(int)mean * 2];
                for (int i = 0; i < (int)mean * 2; i++) {
                    arr[i] = i;
                }
                String arrayString = ArrayUtils.toString(arr);
                arrayString = arrayString.substring(1, arrayString.length() - 1);
                ret = new Pair<>(String.format("[%1$s]", arrayString), null);

            default:
                System.out.println("Warning: cannot infer the bounded support for " + funcId + ". Adaptive algorithm applied automatically.");
                break;
        }

        // if the variable is no longer with flex Support e.g. x has upper lower bound but is a flip
        if (!isFlex){
            varsWithFlexSupport.remove(lhs);
        }

        // if number of parameters is not set, report error
        if (num_params == 0){
            return ret;
        }

        String llhs = lhs;
        if (lhs.contains(".")){
            llhs = lhs.substring(0, lhs.indexOf('.'));
        }

        // case 1: re-assign prior (under the same distribution)
        //
        // float A
        // A = bernoulli(0.5)
        // A = bernoulli(0.6) <--- getBoundsFromPrior()

        // case 2: param is proper (has upper and lower bound), and randvar has NOT been assigned!
        //
        // @limits<lower=-10, upper=10>
        // float C
        // C = bernoulli(0.5) <--- getBoundsFromPrior()

        // case 3: param is improper (lacks bound until assign) and not assigned
        //
        // float B
        // B = bernoulli(0.5) <--- getBoundsFromPrior()


        // case 1
        if (assigned.contains(llhs)){
            List<Integer> params = var2NoiseIndices.get(llhs);
            for (int i = paramIndex; i < paramIndex + num_params; i++){
                params.add(i);
            }
//            param2NoiseIndex.put(llhs, params);
            // update priorPriorDims
            properPriorDims.put(llhs, var2NoiseIndices.size() - 1);

            paramIndex += num_params;

        }
        // case 2
        else if (!assigned.contains(llhs) && properPriorDims.containsKey(llhs)){
            // reset prior index
            resetProperPriorIndex(lhs, num_params);

            // remove noiseBound of the proper prior
            // e.g. noiseBound["C"] = [[-0.1,0.1], [-0.1,0.1]]
            // needs to pop the first 2 entries out
            noiseBound.get(lhs).remove(0);
            noiseBound.get(lhs).remove(0);
        }
        // cases 3
        else{
            // update param2NoiseIndex map
            var2NoiseIndices.put(llhs, IntStream.range(paramIndex, paramIndex + num_params)
                    .boxed()
                    .collect(Collectors.toList()));

            // update priorPriorDims
            properPriorDims.put(llhs, var2NoiseIndices.size() - 1);

            paramIndex += num_params;

            // mark the var as assigned
            assigned.add(llhs);
        }

        return ret;
    }

    /*
    private boolean isDistr(String funcId) {
        return distrSet.contains(funcId);
    }
    */

    private Pair<String, String> getLimits(AST.Decl decl) {
        Pair<String, String> lower_upper = null;
        if (decl.annotations.size() > 0) {
            for (AST.Annotation aa : decl.annotations) {
                if (aa.annotationType == AST.AnnotationType.Limits) {
                    AST.Limits aaLimits = (AST.Limits) aa.annotationValue;

                    // if the parameter has a uniform proper prior i.e. upper and lower bounded
                    if (aaLimits.lower != null && aaLimits.upper != null) {
                        lower_upper = new Pair<>(aaLimits.lower.toString(), aaLimits.upper.toString());
                    }
                    // if bound is (0, inf), improper prior
                    else if (aaLimits.lower != null) {
                        lower_upper = null;
                        sigmaSet.put(decl.id.id, String.valueOf(Double.valueOf(aaLimits.lower.toString()) + 0.0001));
                    }
                    else{
                        // improper prior
                    }
                }
            }
        } else {
            lower_upper = null;
        }
        return lower_upper;
    }

    private void visitBlocks(ArrayList<BasicBlock> basicBlocks) {
        Set<BasicBlock> visited = new HashSet<>();
        Queue<BasicBlock> worklist = new LinkedList<>();

        // sense --------------------------------------
        // initialize the num of params of each var that has been instrumented with noise
        for (String var: var2NoiseIndices.keySet()){
            var2NumParamAddedNoise.put(var, 0);
        };
        // --------------------------------------------

        BasicBlock start = basicBlocks.get(0);
        worklist.add(start);
        visited.add(start);
        while (!worklist.isEmpty()) {
            BasicBlock currBB = worklist.poll();
            // deal with if-then-else's join
            if (blockDensityCubeMap.containsKey(currBB)) {
                densityCube_p_branch = blockDensityCubeMap.get(currBB);
                if (joinBlock.containsKey(currBB)) {
                    String densityCube_p_true = densityCube_p_branch + "_true";
                    String densityCube_p_false = densityCube_p_branch + "_false";
                    if (!currBB.getIncomingEdges().containsKey("meetF"))
                        densityCube_p_false = densityCube_p_branch;
                    String theta = joinBlock.get(currBB);
                    theta = String.format("%s.int()", theta);
                    torchCode.append(indent).append(String.format("%1$s = %1$s + torch.log(%4$s * torch.exp(%2$s) + (1 - %4$s) * torch.exp(%3$s))\n", densityCube_p_branch, densityCube_p_true, densityCube_p_false, theta));
                }
            }
            // deal with loop indent
            boolean needsIndent = false;
            if (loopBody.containsKey(currBB)) {
                indent = indent + "    ";
                needsIndent = true;
                inLoop = true;
                // TODO: fix robust
                // now add robust param at the beginning of any loop
                for (String robust_param: robustParams.keySet()) {
                    Pair<String, String> lower_upper = bounds.get(robust_param);
                    String robust_param_i = robust_param; //String.format("%1$s[%2$s-1]", robust_param, loopBody.get(currBB));
                    zeroLowerToEps(lower_upper);
                    torchCode.append(String.format("%4$s%3$s = torch.arange(%1$s, %2$s + ma_eps, step=(%2$s - %1$s)/splits, device=self.device)\n", lower_upper.getKey(), lower_upper.getValue(), robust_param_i, indent));
                    torchCode.append(getReshapeCode(robust_param_i, paramDims));
                }
            }
            translateBlock(currBB);
            if (inLoop && currBB.getOutgoingEdges().containsKey("back")) {
                for (String robust_param: robustParams.keySet()) {
                    // String robust_param_i = String.format("%1$s[%2$s-1]", robust_param, loopBody.get(currBB));
                    torchCode.append(indent).append(String.format("%1$s = torch.logsumexp(%2$s, %3$s, keepdim=True)\n", densityCube_p_branch, densityCube_p_branch, robustParams.get(robust_param)));
                }
                indent = indent.substring(0, indent.length() - 4);;
                inLoop = false;
            }
            if (currBB.getOutgoingEdges().size() > 0) {
                for(Map.Entry<String, BasicBlock> edge_b:currBB.getOutgoingEdges().entrySet()){
                    BasicBlock b = edge_b.getValue();
                    String edge = edge_b.getKey();
                    if (!visited.contains(b)) {
                        if (edge == null) {
                            worklist.add(b);
                            visited.add(b);
                        }
                        else { // deal with if-then-else
                            if (edge.equals("true") && !loopBody.containsKey(b)) {
                                worklist.add(b);
                                visited.add(b);
                                String densityCube_p_branch_true = densityCube_p_branch + "_true";
                                blockDensityCubeMap.put(b, densityCube_p_branch_true);
                                torchCode.append(indent).append(densityCube_p_branch_true).append(" = torch.tensor(0)\n");
                                BasicBlock join = b.getOutgoingEdges().get("meetT");
                                if (join != null) {
                                    joinBlock.put(join, ifCondStack.pop());
                                    blockDensityCubeMap.put(join, densityCube_p_branch);
                                }
                                else {
                                    join = findJoinBlock(b);
                                    joinBlock.put(join, ifCondStack.pop());
                                    blockDensityCubeMap.put(join, densityCube_p_branch);
                                }
                            } else if (edge.equals("false") && !joinBlock.containsKey(b) && !currBB.getIncomingEdges().containsKey("back")) {
                                worklist.add(b);
                                visited.add(b);
                                String densityCube_p_branch_false = densityCube_p_branch + "_false";
                                blockDensityCubeMap.put(b, densityCube_p_branch_false);
                                torchCode.append(indent).append(densityCube_p_branch_false).append(" = torch.tensor(0)\n");
                            } else if (edge.equals("meetT")) {
                                // blocks above meetT and meetF must be visited,
                                // then we can add the join block to worklist
                                if (visited.contains(b.getIncomingEdges().get("meetF"))) {
                                    worklist.add(b);
                                    visited.add(b);
                                }
                            } else if (edge.equals("meetF")) {
                                if (visited.contains(b.getIncomingEdges().get("meetT"))) {
                                    worklist.add(b);
                                    visited.add(b);
                                }
                            } else {
                                worklist.add(b);
                                visited.add(b);
                            }
                        }
                    }
                }
            }
        }
    }

    // TODO: use a sound analysis to find sigma bounds > 0
    private void zeroLowerToEps(Pair<String, String> lower_upper) {
        // if (lower_upper.getKey().equals("0")) {
        //     lower_upper.setKey("ma_eps");
        // }
    }

    private BasicBlock findJoinBlock(BasicBlock b) {
        int trueCount = 1;
        while (b.getOutgoingEdges().containsKey("true")) {
            trueCount += 1;
            b = b.getOutgoingEdges().get("true");
        }
        while (trueCount > 0) {
            b = b.getOutgoingEdges().get("meetT");
            trueCount -= 1;
        }
        assert b != null;
        return b;
    }

    private void translateBlock(BasicBlock basicBlock) {
        if(basicBlock.getStatements().size() == 0)
            return;
        for (Statement statement : basicBlock.getStatements()){
            if (statement.statement instanceof AST.AssignmentStatement){
                AST.AssignmentStatement assignmentStatement = (AST.AssignmentStatement) statement.statement;
                String lhs = pytorchVisitor.evaluate(assignmentStatement.lhs);
                if (paramDims.containsKey(lhs + "[0]")) {
                    // for vectorized param prior: mu ~ normal(0,1), mu.size() = 3
                    int dim = paramVecLength.get(lhs);
                    for (int i =0;i<dim; i++) {
                        String lhs_i = String.format("%1$s[%2$s]", lhs, i);
                        addLogLik(assignmentStatement, lhs_i);
                    }
                }
                else { // singleton param and "target"
                    addLogLik(assignmentStatement, lhs);
                }
            } else if(statement.statement instanceof AST.Decl) {
                AST.Decl decl = (AST.Decl) statement.statement;
                if (decl.dtype.primitive == AST.Primitive.VECTOR ||
                        (decl.dtype.primitive == AST.Primitive.FLOAT && decl.dims != null)
                        ) {
                    int dim = getDim(decl);
                    String ddid = decl.id.id;
                    paramVecLength.put(ddid, dim);
                    if (!robustParams.containsKey(decl.id.id)) {
                        // for beta with 2d
                        endCode.append(indent).append(String.format("%1$s = [None] * %2$s\n", ddid, dim));
                        endCode.append(indent).append(String.format("posterior_%1$s = [None] * %2$s\n", ddid, dim));
                        if (getDataGrad)
                            endCode.append(indent).append(String.format("expectation_%1$s = [None] * %2$s\n", ddid, dim));
                        findboundCode.append(indent).append(String.format("%1$s = [None] * %2$s\n", ddid, dim));
                        findboundCode.append(indent).append(String.format("posterior_%1$s = [None] * %2$s\n", ddid, dim));
                        findboundCode.append(indent).append(String.format("all_gt_%1$s = [None] * %2$s\n", ddid, dim));
                        findboundCode.append(indent).append(String.format("lowProb_%1$s = [None] * %2$s\n", ddid, dim));
                        // if (adaptive_bounds.containsKey(ddid + "[0]")) {
                        findboundCode.append(indent).append(String.format("adaptiveLower_%1$s = [None] * %2$s\n", ddid, dim));
                        findboundCode.append(indent).append(String.format("adaptiveUpper_%1$s = [None] * %2$s\n", ddid, dim));
                        torchBeginCode.append(indent).append(String.format("%1$s = [None] * %2$s\n", ddid, dim));
                        torchBeginCode.append(indent).append(String.format("posterior_%1$s = [None] * %2$s\n", ddid, dim));
                        torchBeginCode.append(indent).append(String.format("adaptiveUpper_%1$s = [None] * %2$s\n", ddid, dim));
                        torchBeginCode.append(indent).append(String.format("adaptiveLower_%1$s = [None] * %2$s\n", ddid, dim));
                        // }
                        for (int i = 0; i < dim; i++) {
                            String id = String.format("%1$s[%2$s]", decl.id.id, i);
                            getSplits(id);
                        }
                    }
                } else {
                    String id = decl.id.id;
                    getSplits(id);
                    // TODO: fix local params that are not with prefix robust_local_
                    if (inLoop) {
//                        robustParams.put(id, paramDims.get(id));
//                        paramDims.remove(id);
                    }
                }
            } else if (statement.statement instanceof AST.ForLoop) {
                AST.ForLoop forLoop = (AST.ForLoop) statement.statement;
                torchCode.append(indent).append(String.format("for %1$s in range(%2$s, %3$s + 1):\n",
                        forLoop.loopVar.id, forLoop.range.start.toString(), forLoop.range.end.toString()));
                loopBody.put(basicBlock.getOutgoingEdges().get("true"), forLoop.loopVar.id);
            } else if(statement.statement instanceof AST.IfStmt) {
                AST.IfStmt ifStmt = (AST.IfStmt) statement.statement;
                // sense : conditional expression makes no change to the state of the program, thus no need to update boundNoise
                ifCondStack.push("(" + pytorchVisitor.evaluate(ifStmt.condition) + ")");
            } else if(statement.statement instanceof AST.FunctionCallStatement) {
                AST.FunctionCallStatement functionCallStatement = (AST.FunctionCallStatement) statement.statement;
                String funcId = functionCallStatement.functionCall.id.id;
                if (funcId.equals("observe") || funcId.equals("hardObserve")) {
                    pytorchVisitor.update_evaluate(var2NoiseIndices, noiseBound, var2NumParamAddedNoise);
                    String mean = pytorchVisitor.evaluate(functionCallStatement.functionCall.parameters.get(0));
                    if (mean.matches("-?\\d+(\\.\\d+)?")) {
                        torchCode.append(indent).append(String.format("%2$s = %2$s + torch.log(" +
                                "torch.tensor([%1$s], device=self.device))\n", mean, densityCube_p_branch));
                    }
                    else {
                        torchCode.append(indent).append(String.format("%2$s = %2$s + torch.log((%1$s).float())\n", mean, densityCube_p_branch));
                    }
                }
            }
        }
    }

    private void addLogLik(AST.AssignmentStatement assignmentStatement, String lhs) {
        String rhsCode;
        pytorchVisitor.lhs = lhs;
        pytorchVisitor.assignmentStatement = assignmentStatement;
        pytorchVisitor.update_evaluate(var2NoiseIndices, noiseBound, var2NumParamAddedNoise);
        rhsCode = pytorchVisitor.evaluate(assignmentStatement.rhs);
        pytorchVisitor.lhs = null;
        pytorchVisitor.assignmentStatement = null;

        String llhs = lhs;
        if (llhs.contains("[")){
            llhs = llhs.substring(0, llhs.indexOf('['));
            llhs = llhs + "[0]";
        }

        if (((bounds.containsKey(lhs) && bounds.get(lhs) == null ||
                (bounds.containsKey(llhs) && bounds.get(llhs) == null)))) {
            // dependent assign
            torchCode.append(indent).append(lhs).append("=").append(rhsCode).append("\n");
        } else { // for param and "target"
            torchCode.append(indent).append(String.format("%1$s = %1$s + ", densityCube_p_branch)).append(rhsCode).append("\n");
        }
    }

    private void getSplits(String id) {
        if (!bounds.containsKey(id)) {
        // no id in bounds, it's a param without prior
            Pair<String, String> rhs_bounds = new Pair<>(
                    String.format("adaptiveLower_%1$s",id),
                    String.format("adaptiveUpper_%1$s",id));
            bounds.put(id, rhs_bounds);
            String rawId = id.split("\\[", 2)[0];
            if (!adaptive_bounds.containsKey(id)) // otherwise the init bounds are set in processDecl
                adaptive_bounds.put(id, new Pair<>(sigmaSet.getOrDefault(rawId, "-50"), "50"));
        }
        Pair<String, String> lower_upper = bounds.get(id);
        if (lower_upper != null) {
            if (lower_upper.getValue() != null) { // generate uniform splits
                zeroLowerToEps(lower_upper);

                // sense corner case here
                // 1) if noise affects support of variable, e.g. uniform(0 + noise[i], 1 + noise[i + 1]), bounds needs to be updated
                // 2) otherwise no change
                String lo = lower_upper.getKey();
                String up = lower_upper.getValue();

                if (varsWithFlexSupport.containsKey(id)){
                    String loNoise = String.format("(%1$s + noise[%2$s])", lo , var2NoiseIndices.get(id).get(0));
                    String upNoise = String.format("(%1$s + noise[%2$s])", up , var2NoiseIndices.get(id).get(1));

                    torchCode.append(String.format("%3$sstep = ((%2$s - %1$s) / splits).repeat([splits + 1])\n", loNoise, upNoise, indent));
                    torchCode.append(String.format("%1$sints = torch.arange(0, splits + ma_eps, step = 1, device=self.device)\n", indent));
                    torchCode.append(String.format("%2$s%3$s = torch.add((step * ints), %1$s)\n", loNoise, indent, id));
                }
                else{
                    torchCode.append(String.format("%4$s%3$s = torch.arange(%1$s, %2$s + ma_eps, step=(%2$s - %1$s)/splits, device=self.device)\n",
                            lower_upper.getKey(), lower_upper.getValue(), id, indent));
                }

                torchCode.append(getReshapeCode(id, paramDims));
            } else { // splits are given, e.g., bernoulli ==> 0, 1
                if (varsWithFlexSupport.containsKey(id)){
                    String distType = varsWithFlexSupport.get(id);
                    switch (distType){
                        case "uniformInt":{
                            String set = lower_upper.getKey(); // e.g. [2,3,4]
                            ArrayList<Integer> commas = new ArrayList<>();
                            for (int i = 0; i < set.length(); ++i){
                                if (set.charAt(i) == ','){
                                    commas.add(i);
                                }
                            }

                            String loNoise = String.format("(%1$s + noise[%2$s])", set.substring(1,commas.get(0)), var2NoiseIndices.get(id).get(0));
                            String upNoise = String.format("(%1$s + noise[%2$s])", set.substring(commas.get(commas.size() - 1) + 1,set.length() - 1), var2NoiseIndices.get(id).get(1));

                            torchCode.append(String.format("%4$s%3$s = torch.range(%1$s, %2$s, step = 1, device=self.device)\n", loNoise, upNoise, id, indent));
                            break;
                        }
                        case "atom" :{
                            String atom = lower_upper.getKey().substring(1, lower_upper.getKey().length() - 1); // e.g. [123]
                            String n = String.format("[(%1$s + noise[%2$s])]", atom, var2NoiseIndices.get(id).get(0));
                            torchCode.append(String.format("%3$s%1$s = torch.tensor(%2$s, device=self.device)\n", id, n, indent));
                            break;
                        }
                        case "discrete":    {
                            String set = lower_upper.getKey(); // e.g. [2,3,4]
                            List<String> s = Arrays.asList(set.substring(1, set.length() - 1).split(","));
                            List<String> s2 = new ArrayList<>();
                            for (int i = 0; i < s.size(); i++) {
                                s2.add(String.format("(%1$s + noise[%2$s]),", s.get(i), var2NoiseIndices.get(id).get(i)));
                            }
                            String n = "[" + s2.stream().reduce((pn1, pn2) -> pn1 + pn2).orElse("") + "]";
                            torchCode.append(String.format("%3$s%1$s = torch.tensor(%2$s, device=self.device)\n", id, n, indent));
                            break;
                        }
                    }
                }
                else{
                    torchCode.append(String.format("%3$s%1$s = torch.tensor(%2$s, device=self.device)\n", id, lower_upper.getKey(),indent));
                }
                torchCode.append(getReshapeCode(id, paramDims));
            }
        } // otherwise it is dependent var, so no init
    }

    private void getDataString(ArrayList<AST.Data> dataSets) {
        for (AST.Data dd: dataSets) {
            if (dd.expression != null) {
                String ddid = dd.decl.id.id;
                dataSection.append(indent).append(String.format("%1$s = %2$s\n", ddid, dd.expression.toString()));
            }
            else if (dd.array != null) {
                String ddid = dd.decl.id.id;
                dataSection.append(indent).append(String.format("%1$s = torch.tensor(%2$s, device=self.device)\n", ddid, dd.array.expressions));
                dataSection.append(getReshapeCode(ddid, dataDims));
                // if (getDataGrad) {
                //     dataSection.append(indent).append(String.format("%1$s.requires_grad = True\n", ddid));
                //     dataSection.append(indent).append(String.format("%1$s.retain_grad()\n", ddid));
                // }
            }
            else if (dd.vector != null) {
                String ddid = dd.decl.id.id;
                dataSection.append(indent).append(String.format("%1$s = torch.tensor(%2$s, device=self.device)\n", ddid, dd.vector.expressions));
                dataSection.append(getReshapeCode(ddid, dataDims));
                // if (getDataGrad) {
                //     dataSection.append(indent).append(String.format("%1$s.requires_grad = True\n", ddid));
                //     dataSection.append(indent).append(String.format("%1$s.retain_grad()", ddid));
                // }
            }
        }
    }

    private String getReshapeCode(String ddid, Map<String, Integer> paramMap) {
        int curr_param_dim;
        if (paramMap.containsKey(ddid)) {
            curr_param_dim = paramMap.get(ddid);
        }
        else {
            curr_param_dim = robustParams.get(ddid.split("\\[", 2)[0]);
        }
        shape[curr_param_dim] = -1;
        String ret = String.format("%3$s%1$s = torch.reshape(%1$s, %2$s)\n", ddid, Arrays.toString(shape), indent);
        shape[curr_param_dim] = 1;
        return ret;
    }

    public String fromStan(String stanPath) {
        int index0=stanPath.lastIndexOf('/');
        String stanName = stanPath.substring(index0+1,stanPath.length());
        String stanfile = stanPath + "/" + stanName + ".stan";
        String standata = stanPath + "/" + stanName + ".data.R";
        Stan2IRTranslator stan2IRTranslator = new Stan2IRTranslator(stanfile, standata);
        String tempFileName = stanfile.replace(".stan", "");
        String templateCode = stan2IRTranslator.getCode();

        // System.out.println("========Stan Code to Template=======");
        // System.out.println(templateCode);

        File tempfilePath = null;
        try {
            tempfilePath = File.createTempFile(tempFileName, ".template");
            // tempfilePath = new File(stanPath + "/" + stanName + ".template");
            FileUtils.writeStringToFile(tempfilePath, templateCode);
        } catch (IOException e) {
            e.printStackTrace();
        }
        assert tempfilePath != null;
        return runCompiler(tempfilePath.getAbsolutePath());
    }


    static {
        try {
            String CONFIGURATIONFILE = "src/main/resources/aqua2torch.properties";
            FileInputStream fileInputStream = new FileInputStream(CONFIGURATIONFILE);
            properties.load(fileInputStream);
            sameInnerSplits = Boolean.parseBoolean(properties.getProperty("sameInnerSplits"));
            adaptive = Boolean.parseBoolean(properties.getProperty("adaptive"));
            enableTiming = Boolean.parseBoolean(properties.getProperty("enableTiming"));
            getCFGPlot = Boolean.parseBoolean(properties.getProperty("getCFGPlot"));
            getDataGrad = Boolean.parseBoolean(properties.getProperty("getDataGrad"));
            splits = properties.getProperty("split");
        } catch (IOException var2) {
            var2.printStackTrace();
        }

    }
}
