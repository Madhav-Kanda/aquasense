package aqua.analyses;

import grammar.AST;
import grammar.analyses.Pair;
import translators.visitors.BaseVisitor;

import java.util.*;
import java.util.stream.Collectors;


class Bound {
    public Bound(float f){
        this.f = f;
    }

    public Bound(String s){
        this.s = s;
    }

    public Bound(Pair<Bound, Bound> p) {this.symbolic = p;}
    float f;
    String s;
    Pair<Bound, Bound> symbolic;
}
public class PytorchVisitor extends BaseVisitor {
    String lhs = null;
    AST.AssignmentStatement assignmentStatement = null;
    Map<String, Integer> dataDims;
    HashMap<String, Integer> robustParams;

    // sense --------------------------------------------------------------------
    Pair<Number, Number> defaultNoiseBound = new Pair<>(-0.1,0.1);
    Pair<Number, Number> defaultNoiseBoundPos = new Pair<>(0.,0.1);

    Double noiseRatioLo = -0.1;
    Double noiseRatioHg = 0.1;


    private Map<String, List<Integer>> var2NoiseIndices;
    private Map<String, ArrayList<Pair<Number, Number>>> noiseBound;

    private Map<String, Integer> var2NumParamAddedNoise = new HashMap<>();

    public void update_evaluate(Map<String, List<Integer>> var2NoiseIndices, Map<String, ArrayList<Pair<Number, Number>>> noiseBound, Map<String, Integer> var2NumParamAddedNoise){
        this.var2NoiseIndices = var2NoiseIndices;
        this.noiseBound = noiseBound;
        this.var2NumParamAddedNoise = var2NumParamAddedNoise;
    }

    // ---------------------------------------------------------------------------

    @Override
    public String evaluate(AST.ArrayAccess arrayAccess) {
        String aaid = arrayAccess.id.id;
        if (!dataDims.containsKey(aaid)) {
            // TODO: currently let robust_local_ appear without array access
            if (robustParams.containsKey(aaid)) {
                return aaid;
            } else {
                String dim_str = evaluate(arrayAccess.dims.dims.get(0));
                if (!dim_str.matches("-?\\d+(\\.\\d+)?")) {
                    return aaid + "[" + dim_str + "-1]";
                } else {
                    return aaid + "[" + String.valueOf(Integer.valueOf(dim_str) - 1) + "]";
                }
            }
        }
        else {
            String dim_str = evaluate(arrayAccess.dims.dims.get(0));
            return aaid + ".view(-1)" + "[" + dim_str + "-1]";
        }

    }

    public String evaluate(AST.FunctionCall functionCall) {
        String funcId = functionCall.id.id;
        String ret = "";

        // if the lhs is an array access
        // e.g.
        // for(i in 1:N){
        // @observe
        //  y[i] = normal(a[county[i]],sigma_y)
        // }
        // lhs would be "y.view(-1)[i-1]"
        // lhs needs to be truncated to "y"
        String llhs = lhs;
        if (lhs.contains(".")){
            llhs = lhs.substring(0, lhs.indexOf('.'));
        }

        // if the statement is OBSERVE, no NOISE instrumentation
        boolean isObserve = dataDims.containsKey(llhs) || llhs.equals("densityCube_p");


        // if the lhs is a vectorized parameter
        // e.g.
        // for (i in 1:N) {
        //     mu[i] = normal(0,1)
        // }
        // lhs would be mu[i-1]
        // obtain the paramIndex of mu[0], then calculate the correct paramIndex = paramIndex[mu[0]].first + params(mu) * index

        ArrayList<String> paramIndex = new ArrayList<>();


        if (lhs != "densityCube_p" && !isObserve){
            try {
                paramIndex = (ArrayList<String>) var2NoiseIndices.get(llhs)
                        .subList(var2NumParamAddedNoise.get(llhs), var2NoiseIndices.get(llhs).size())
                        .stream()
                        .map(String::valueOf)
                        .collect(Collectors.toList());
            } catch (NullPointerException e) {
                if (llhs.contains("[")){
                    llhs = llhs.substring(0, llhs.indexOf('['));
                }
                paramIndex = new ArrayList<String>();
                for (int add = 0; add < var2NoiseIndices.get(llhs + "[0]").size(); ++add){
                    String index = lhs.substring(lhs.indexOf('[') + 1, lhs.lastIndexOf(']'));
                    int base = var2NoiseIndices.get(llhs + "[0]").get(0);
                    int size = var2NoiseIndices.get(llhs + "[0]").size();
                    paramIndex.add(String.format("%1$s + (%2$s) * %3$s + %4$s", base, index, size, add));
                }
            }
        }

        int num_noise = 0;
        ArrayList<Pair<Number, Number>> oldNoiseBounds = noiseBound.get(llhs);
        ArrayList<Pair<Number, Number>> newNoiseBounds = null;

        if (lhs != null) {
            switch (funcId) {
                case "normal": {
                    num_noise = 2;

                    String sd = evaluate(functionCall.parameters.get(1));
                    String mean = evaluate(functionCall.parameters.get(0));

                    if (!isObserve){
                        Pair<Number, Number> Bound1 = defaultNoiseBound;
                        Pair<Number, Number> Bound2 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(mean);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(sd);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));
                        mean = String.format("(%1$s + noise[%2$s])", mean , paramIndex.get(0));
                        sd = String.format("(%1$s + noise[%2$s])", sd , paramIndex.get(1));
                    }

                    ret = getNormalDistr(sd, mean, lhs);
                    break;
                }
                case "normal_lpdf": {
                    // log likelihood of normal distribution seeing the data
                    String sd = evaluate(functionCall.parameters.get(2));
                    String mean = evaluate(functionCall.parameters.get(1));
                    String data = evaluate(functionCall.parameters.get(0));

                    ret = getNormalDistr(sd, mean, data);
                    break;
                }
                case "gamma": {
                    num_noise = 2;
                    String alpha = evaluate(functionCall.parameters.get(0));
                    String beta = evaluate(functionCall.parameters.get(1));

                    if (!isObserve){
                        Pair<Number, Number> Bound1 = new Pair<>(-3, 3);
                        Pair<Number, Number> Bound2 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(alpha);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(beta);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));

                        alpha = String.format("(%1$s + noise[%2$s])", alpha , paramIndex.get(0));
                        beta = String.format("(%1$s + noise[%2$s])", beta , paramIndex.get(1));
                    }
                    ret = String.format("torch.nan_to_num(tdist.Gamma(%1$s, %2$s).log_prob(%3$s).to(self.device), nan=-float('inf'), posinf=0, neginf=0)",
                            alpha, beta, lhs);

                    break;
                }
                case "beta": {
                    num_noise = 2;
                    String alpha = evaluate(functionCall.parameters.get(0));
                    String beta = evaluate(functionCall.parameters.get(1));

                    if (!isObserve){
                        Pair<Number, Number> Bound1 = new Pair<>(-1, 1);
                        Pair<Number, Number> Bound2 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(alpha);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(beta);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>((Collection<? extends Pair<Number, Number>>) Arrays.asList(Bound1, Bound2));

                        alpha = String.format("(%1$s + noise[%2$s])", alpha , paramIndex.get(0));
                        beta = String.format("(%1$s + noise[%2$s])", beta , paramIndex.get(1));
                    }
                    ret = String.format("torch.nan_to_num(tdist.Beta(%1$s, %2$s).log_prob(%3$s).to(self.device), nan=-float('inf'), posinf=0, neginf=0)",
                            alpha, beta, lhs);

                    break;
                }
                case "categorical": {
                    num_noise = functionCall.parameters.size();
                    int num_params = functionCall.parameters.size();

                    StringBuilder params = new StringBuilder();
                    newNoiseBounds = new ArrayList<Pair<Number, Number>>();

                    params.append('[');
                    for (int i = 0; i < num_params; ++i){
                        String next;
                        if (!isObserve){
                            next = String.format("%1$s + noise[%2$s],", evaluate(functionCall.parameters.get(i)) ,paramIndex.get(i));
                        } else{
                            next = String.format("%1$s,", evaluate(functionCall.parameters.get(i)));
                        }
                        params.append(next);
                        newNoiseBounds.add(new Pair<>(-.1, 0.1));
                    }
                    params.append(']');

                    ret = String.format("torch.log(torch.tensor(%1$s, device=self.device).reshape(%2$s.shape))", params, lhs);
                    break;
                }
                case "poisson": {
                    num_noise = 1;
                    String mean = evaluate(functionCall.parameters.get(0));

                    if (!isObserve) {
                        Pair<Number, Number> Bound1 = new Pair<>(-3, 3);
                        try {
                            double p1 = Double.parseDouble(mean);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));

                        mean = String.format("(%1$s + noise[%2$s])", mean , paramIndex.get(0));
                    }

                    ret = String.format("tdist.Poisson(%1$s).log_prob(%2$s).to(self.device)", mean, lhs);
                    break;
                }
                case "discrete": {
                    ret = String.format("torch.log(torch.tensor(1/ (%1$s), device=self.device))", functionCall.parameters.size());
                    break;
                }
                case "student_t": {
                    //TODO add noise
                    String loc = evaluate(functionCall.parameters.get(1));
                    String scale = evaluate(functionCall.parameters.get(2));
                    String df = evaluate(functionCall.parameters.get(0));
                    ret = getStudTLpdf(loc, scale, df, lhs);
                    break;
                }
                case "binomial": {
                    num_noise = 2;
                    String total_count = evaluate(functionCall.parameters.get(0));
                    String p = evaluate(functionCall.parameters.get(1));

                    if (!isObserve) {
                        Pair<Number, Number> Bound1 = new Pair<>(-3, 3);
                        Pair<Number, Number> Bound2 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(total_count);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(p);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));

                        total_count = String.format("(%1$s + noise[%2$s])", total_count , paramIndex.get(0));
                        p = String.format("(%1$s + noise[%2$s])", p , paramIndex.get(1));
                    }

                    ret = String.format("tdist.Binomial(%1$s, %2$s).log_prob(%3$s).to(self.device)", total_count, p, lhs);

                    break;
                }
                case "student_t_lpdf": {
                    String data = evaluate(functionCall.parameters.get(0));
                    String df = evaluate(functionCall.parameters.get(1));
                    String loc = evaluate(functionCall.parameters.get(2));
                    String scale = evaluate(functionCall.parameters.get(3));
                    ret = getStudTLpdf(loc, scale, df, data);
                    break;
                }
                case "uniform": {
                    num_noise = 2;
                    AST.Expression loExpr = functionCall.parameters.get(0);
                    AST.Expression upExpr = functionCall.parameters.get(1);
                    String lo = evaluate(loExpr);
                    String up = evaluate(upExpr);

                    if (!isObserve) {
                        String loNoise = String.format("(%1$s + noise[%2$s])", lo , paramIndex.get(0));
                        String upNoise = String.format("(%1$s + noise[%2$s])", up , paramIndex.get(1));

                        if (loExpr instanceof AST.UnaryExpression)
                            loExpr = ((AST.UnaryExpression) loExpr).expression;
                        if (upExpr instanceof AST.UnaryExpression)
                            upExpr = ((AST.UnaryExpression) upExpr).expression;
                        if (loExpr instanceof AST.Number && upExpr instanceof AST.Number)
                            ret = String.format("torch.log(1/ (%2$s - %1$s) * torch.logical_and(%3$s >= %1$s, %3$s <= %2$s))", loNoise, upNoise, lhs);
                        else
                            ret = String.format("torch.log(torch.nan_to_num(1/ (%2$s - %1$s), posinf=0, neginf=0) * torch.logical_and(%3$s >= %1$s, %3$s <= %2$s))", loNoise, upNoise, lhs);


                        Pair<Number, Number> Bound1 = new Pair<>(-1, 1);
                        Pair<Number, Number> Bound2 = new Pair<>(-1, 1);

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

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));
                    }

                    ret = String.format("torch.log(1/ (%2$s - %1$s) * torch.logical_and(%3$s >= %1$s, %3$s <= %2$s))", lo, up, lhs);
                    break;
                }
                case "uniformClose": {
                    // TODO add noise
                    String lo = evaluate(functionCall.parameters.get(0));
                    String up = evaluate(functionCall.parameters.get(1));

                    if (!isObserve) {
                        Pair<Number, Number> Bound1 = new Pair<>(-1, 1);
                        Pair<Number, Number> Bound2 = new Pair<>(-1, 1);

                        try {
                            double p1 = Double.parseDouble(lo);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(up);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));
                    }
                    ret = String.format("torch.log((1/ (%2$s - %1$s)) * torch.logical_and(%3$s >= %1$s, %3$s <= %2$s))",
                            lo, up, lhs);

                    break;
                }
                case "uniformInt": {
                    num_noise = 2;
                    String lo = evaluate(functionCall.parameters.get(0));
                    String up = evaluate(functionCall.parameters.get(1));

                    if (!isObserve) {
                        lo = String.format("(%1$s + noise[%2$s])", lo , paramIndex.get(0));
                        up = String.format("(%1$s + noise[%2$s])", up , paramIndex.get(1));

                        Pair<Number, Number> Bound1 = new Pair<>(-1, 1);
                        Pair<Number, Number> Bound2 = new Pair<>(-1, 1);

                        try {
                            double p1 = Double.parseDouble(lo);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}
                        try {
                            double p2 = Double.parseDouble(up);
                            if (p2 != 0){
                                Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2));
                    }
                    ret = String.format("torch.log(1/(%2$s - %1$s + 1) * torch.logical_and(%3$s >= %1$s,%3$s <= %2$s))",
                            lo, up, lhs);
                    break;
                }
                case "flip": {
                    num_noise = 1;
                    String p = evaluate(functionCall.parameters.get(0));

                    if (!isObserve) {
                        p = String.format("(%1$s + noise[%2$s])",p , paramIndex.get(0));

                        Pair<Number, Number> Bound1 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(p);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));
                    }
                    ret = String.format("torch.log((%1$s) * (%2$s) + (1 - (%1$s)) * (1 - (%2$s)))", p, lhs);
                    break;
                }
                case "bernoulli": {
                    num_noise = 1;
                    String p = evaluate(functionCall.parameters.get(0));
                    if (!isObserve){
                        p = String.format("(%1$s + noise[%2$s])",p , paramIndex.get(0));

                        Pair<Number, Number> Bound1 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(p);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));
                    }
                    ret = String.format("torch.log((0 <= %2$s).int() * (1 >= %2$s).int() * ((%1$s) * (%2$s) + (1 - (%1$s)) * (1 - (%2$s))))", p, lhs);

                    break;
                }
                case "geometric": {
                    num_noise = 1;
                    String p = evaluate(functionCall.parameters.get(0));

                    if (!isObserve) {
                        p = String.format("(%1$s + noise[%2$s])",p , paramIndex.get(0));

                        Pair<Number, Number> Bound1 = defaultNoiseBound;

                        try {
                            double p1 = Double.parseDouble(p);
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));
                    }

                    ret = String.format("torch.nan_to_num(tdist.Geometric(%1$s).log_prob(%2$s).to(self.device), nan=-float('inf'), posinf=0, neginf=0)", p, lhs);
                    break;
                }
                case "bernoulli_logit": {
                    num_noise = 1;
                    String alpha = evaluate(functionCall.parameters.get(0));

                    if (!isObserve){
                        alpha = String.format("(%1$s + noise[%2$s])",alpha , paramIndex.get(0));

                        Pair<Number, Number> Bound1 = defaultNoiseBound;

                        try {
                            double a1 = Double.parseDouble(alpha);
                            if (a1 != 0){
                                Bound1 = new Pair<>(a1 * noiseRatioLo, a1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));
                    }

                    ret = getBernoulliLogitLpmf(alpha, lhs);

                    break;
                }
                case "bernoulli_logit_lpmf": {
                    String data = evaluate(functionCall.parameters.get(0));
                    String alpha = evaluate(functionCall.parameters.get(1));
                    ret = getBernoulliLogitLpmf(alpha, data);
                    break;
                }
                case "atom": {
                    num_noise = 1;
                    String atom = evaluate(functionCall.parameters.get(0));
                    if (!isObserve) {
                        atom = String.format("(%1$s + noise[%2$s])", evaluate(functionCall.parameters.get(0)), paramIndex.get(0));

                        Pair<Number, Number> Bound1 = new Pair<>(-1, 1);

                        try {
                            double p1 = Double.parseDouble(evaluate(functionCall.parameters.get(0)));
                            if (p1 != 0){
                                Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                            }
                        } catch (NumberFormatException ignored) {}

                        newNoiseBounds = new ArrayList<>(Collections.singletonList(Bound1));
                    }
                    ret = String.format("torch.log(%1$s == %2$s)", lhs, atom);
                    break;
                }
                case "triangle": {
                    // TODO add noise
                    String m = evaluate(functionCall.parameters.get(0));
                    String l = evaluate(functionCall.parameters.get(1));
                    String r = evaluate(functionCall.parameters.get(2));
                    ret = String.format("torch.log(torch.logical_and(%4$s>=%1$s-(%2$s),%4$s<=%1$s)*2*(%4$s-(%1$s-(%2$s)))/(%2$s*(%2$s+%3$s)) + torch.logical_and(%4$s>%1$s,%4$s<=%1$s+%3$s)*2*((%1$s+%3$s)-(%4$s))/(%3$s*(%2$s+%3$s)))", m, l, r, lhs);

                    Pair<Number, Number> Bound1 = new Pair<>(-3, 3);
                    Pair<Number, Number> Bound2 = new Pair<>(-3, 3);
                    Pair<Number, Number> Bound3 = new Pair<>(-3, 3);

                    try {
                        double p1 = Double.parseDouble(m);
                        if (p1 != 0){
                            Bound1 = new Pair<>(p1 * noiseRatioLo, p1 * noiseRatioHg);
                        }
                    } catch (NumberFormatException ignored) {}
                    try {
                        double p2 = Double.parseDouble(l);
                        if (p2 != 0){
                            Bound2 = new Pair<>(p2 * noiseRatioLo, p2 * noiseRatioHg);
                        }
                    } catch (NumberFormatException ignored) {}
                    try {
                        double p3 = Double.parseDouble(r);
                        if (p3 != 0){
                            Bound2 = new Pair<>(p3 * noiseRatioLo, p3 * noiseRatioHg);
                        }
                    } catch (NumberFormatException ignored) {}

                    newNoiseBounds = new ArrayList<>(Arrays.asList(Bound1, Bound2, Bound3));
                    break;
                }
                case "exp":
                    ret = String.format("torch.exp(%1$s)", evaluate(functionCall.parameters.get(0)));
                    break;
                case "log": {
                    AST.Expression meanExpr = functionCall.parameters.get(0);
                    String mean = evaluate(meanExpr);
                    if (meanExpr instanceof AST.Number) { // meanExpr > 0
                        ret = String.format("torch.log(torch.tensor([%1$s],device=self.device))", mean);
                    } else {
                        ret = String.format("torch.log(%1$s)", mean);
                    }
                    break;
                }
                case "sqrt": {
                    AST.Expression meanExpr = functionCall.parameters.get(0);
                    String mean = evaluate(meanExpr);
                    if (meanExpr instanceof AST.Number) { // meanExpr > 0
                        ret = String.format("torch.sqrt(torch.tensor([%1$s],device=self.device))", mean);
                    } else {
                        ret = String.format("torch.sqrt(%1$s)", mean);
                    }
                    break;
                }
                case "inv": {
                    ret = String.format("(1 / (%1$s))", evaluate(functionCall.parameters.get(0)));
                    break;
                }
                case "abs": {
                    ret = String.format("torch.abs(%1$s)", evaluate(functionCall.parameters.get(0)));
                    break;
                }
                case "log_mix": {
                    String exp1 = evaluate(functionCall.parameters.get(1));
                    String exp2 = evaluate(functionCall.parameters.get(2));
                    AST.Expression thetaExpr = functionCall.parameters.get(0);
                    String theta = evaluate(thetaExpr);
                    if (thetaExpr instanceof AST.Number) // meanExpr > 0
                        theta = String.format("torch.tensor(%s, device=self.device)", theta);
                    ret = String.format("torch.log(%3$s * torch.exp(%1$s) + (1 - %3$s) * torch.exp(%2$s))",exp1, exp2, theta);
                    break;
                }
                case "fmax": {
                    String param1 = evaluate(functionCall.parameters.get(0));
                    String param2 = evaluate(functionCall.parameters.get(1));
                    ret = String.format("torch.maximum(%1$s, %2$s)", param1, param2);
                    break;
                }
            }
            // for array data, sum up all log probs
            if (dataDims.containsKey(lhs)) {
                ret = String.format("torch.sum(%1$s, %2$d, keepdim=True)", ret, dataDims.get(lhs));
            }

            // concate oldNoiseBounds and newNoiseBounds, store
            if (oldNoiseBounds != null) {
                oldNoiseBounds.addAll(newNoiseBounds);
                noiseBound.put(llhs, oldNoiseBounds);
            } else if (newNoiseBounds != null) {
                noiseBound.put(llhs, newNoiseBounds);
            }
        }

        if (!isObserve){
            var2NumParamAddedNoise.put(llhs, var2NumParamAddedNoise.get(llhs) + num_noise);
        }

        return ret;
    }

    private String getStudTLpdf(String loc, String scale, String df, String lhs) {
        String ret;
        ret = String.format("torch.nan_to_num(tdist.StudentT(%1$s, loc=%2$s, scale=%3$s).log_prob(%4$s).to(self.device), nan=-float('inf'), posinf=0, neginf=0)",
                df, loc, scale, lhs);
        return ret;
    }

    private String getBernoulliLogitLpmf(String alpha, String lhs) {
        String ret;
        ret = String.format("torch.log((2 * %2$s - 1) * torch.sigmoid(%1$s) + (1 - %2$s))",alpha, lhs);
        return ret;
    }

    private String getNormalDistr(String sd, String mean, String lhs) {
        String ret;
        if (sd.matches("-?\\d+(\\.\\d+)?"))
            // TODO : what is this case?
            sd = String.format("torch.tensor(%s, device=self.device)", sd);
        // if (sd.matches("-?\\d+(\\.\\d+)?"))
        //     ret = String.format("- 0.5 * torch.pow((%2$s - (%1$s)) / (%3$s), 2)",
        //             lhs, mean, sd);
        // else
        ret = String.format("torch.nan_to_num(-torch.log(%3$s)- 0.9189385332046727 - 0.5 * torch.pow((%2$s - (%1$s)) / (%3$s), 2), nan=-float('inf'), posinf=0, neginf=0)",
               lhs, mean, sd);
        return ret;
    }

    public String evaluate(AST.Expression expression) {
        if (expression instanceof AST.Id) {
            return this.evaluate((AST.Id)expression);
        } else if (expression instanceof AST.ArrayAccess) {
            return this.evaluate((AST.ArrayAccess)expression);
        } else if (expression instanceof AST.FunctionCall) {
            return this.evaluate((AST.FunctionCall)expression);
        } else if (expression instanceof AST.AddOp) {
            return this.evaluate((AST.AddOp)expression);
        } else if (expression instanceof AST.MinusOp) {
            return this.evaluate((AST.MinusOp)expression);
        } else if (expression instanceof AST.DivOp) {
            return this.evaluate((AST.DivOp)expression);
        } else if (expression instanceof AST.MulOp) {
            return this.evaluate((AST.MulOp)expression);
        } else if (expression instanceof AST.ExponOp) {
            return this.evaluate((AST.ExponOp)expression);
        } else if (expression instanceof AST.UnaryExpression) {
            return this.evaluate((AST.UnaryExpression)expression);
        } else if (expression instanceof AST.Braces) {
            return this.evaluate((AST.Braces)expression);
        } else if (expression instanceof AST.Number) {
            return this.evaluate((AST.Number)expression);
        } else if (expression instanceof AST.VecMulOp) {
            return this.evaluate((AST.VecMulOp)expression);
        } else if (expression instanceof AST.VecDivOp) {
            return this.evaluate((AST.VecDivOp)expression);
        } else if (expression instanceof AST.AndOp) {
            AST.AndOp addOp = (AST.AndOp) expression;
            return String.format("torch.logical_and(%1$s, %2$s).int()", evaluate(addOp.op1), evaluate(addOp.op2));
        } else if (expression instanceof AST.OrOp) {
            AST.OrOp orOp = (AST.OrOp) expression;
            return String.format("torch.logical_or(%1$s, %2$s).int()", evaluate(orOp.op1), evaluate(orOp.op2));
        } else if (expression instanceof AST.TernaryIf) {
            return this.evaluate((AST.TernaryIf)expression);
        } else if (expression instanceof AST.GtOp) {
            return this.evaluate((AST.GtOp)expression);
        } else if (expression instanceof AST.EqOp) {
            return this.evaluate(((AST.EqOp) expression).op1) + " == " + this.evaluate(((AST.EqOp) expression).op2);
        } else if (expression instanceof AST.LeqOp) {
            AST.LeqOp leqOp  = (AST.LeqOp)expression;
            return this.evaluate(leqOp.op1) + "<=" + this.evaluate(leqOp.op2);
        } else if (expression instanceof AST.GeqOp) {
            AST.GeqOp geqOp  = (AST.GeqOp)expression;
            return this.evaluate(geqOp.op1) + ">=" + this.evaluate(geqOp.op2);
        } else {
            //return expression.toString();
            AST.LtOp ltOp = (AST.LtOp)expression;
            return this.evaluate(ltOp.op1) + "<" + this.evaluate(ltOp.op2);

        }
    }
    // @Override
    // public String evaluate(AST.AddOp addOp) {
    //     if (lhs != null && lhs.equals("target")) {

    //
    //     } else {
    //         return evaluate(addOp.op1) + "+" + evaluate(addOp.op2);
    //     }
    // }
    @Override
    public String evaluate(AST.Id id) {
        if (id.id.equals("target")) {
            return "densityCube_p";
        } else {
            return id.id;
        }
    }

    // @Override
    // public String evaluate(AST.Braces braces) {
    //     return "(" + evaluate(braces.expression) + ")";
    // }


    public String evaluate(AST.TernaryIf ternaryIf) {
        String cond = evaluate(ternaryIf.condition);
        String trueExpr = evaluate(ternaryIf.trueExpression);
        String falseExpr = evaluate(ternaryIf.falseExpression);
        return String.format("torch.where(%1$s, %2$s, %3$s)", cond, trueExpr, falseExpr);
    }

    public String evaluate(AST.GtOp gtOp) {
        return evaluate(gtOp.op1) + ">" + evaluate(gtOp.op2);
    }
}


