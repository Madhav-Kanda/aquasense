#!/usr/bin/python3

# benchmark.py
# automatically consturct RQ1~RQ3 results

import json
import logging
import os
import subprocess
import functools
import operator
import itertools
import psutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeout import Command
from functools import partial
from shutil import copyfile, rmtree
from re import T, search, findall, split, match
from scipy.stats import gmean
from scipy import mean

java_bin = "java"
logging.getLogger().setLevel(logging.INFO)
# pd.options.display.float_format = '${:,.2f}'.format

ERR_CODE = -256
TO_CODE = -128

# convergence criteria
DIST_THRES = 1e-6
REL_THRES = 0.05
ABS_THRES = 1e-6

temp_dir = "./benchmarks/psense_bench/"
stan_dir = "./benchmarks/stan_bench/"

global_spl = [100,200,400,800,1600,3200,6400,12800,25600]
global_t = [[] for i in range(9)]
global_abs_err = [[] for i in range(9)]
global_rel_err = [[] for i in range(9)]

model_dists = {
"coins" : {"name": "coins", "Dist.": {"prior":[("bernoulli", 2)], "obs":[]}, "D/C": True, "id":["c1", "c2"]}, 
"explain_away": {"name": "expl\_away", "Dist.": {"prior":[("uniformInt", 2)], "obs":[("uniformInt", 2)]}, "D/C": True, "id":["A", "B"]},
"bayes_rule" : {"name": "bys\_rule", "Dist.": {"prior":[("bernoulli", 3)], "obs":[("bernoulli", 1)]}, "D/C": True, "id":["hypothesis", "data", "data"]},
"murderMystery" : {"name": "murder", "Dist.": {"prior":[("bernoulli", 3)], "obs":[("bernoulli", 1)]}, "D/C": True, "id":["aliceDunnit", "withGun", "withGun"]},
"beta_clinical" : {"name": "beta\_clinic", "Dist.": {"prior":[("bernoulli", 1), ("beta", 3)], "obs":[("bernoulli", 4)]}, "D/C": False, "id":["isEffective", "probIfControl", "probIfTreated", "probIfTreated"]},
"gamma" : {"name": "gamma", "Dist.": {"prior":[("gamma", 1)], "obs":[]}, "D/C": False, "id":["c"]},
"murderMysteryEq": {"name": "murderEq", "Dist.": {"prior":[("bernoulli", 3)], "obs":[("bernoulli", 1)]}, "D/C": True},
"binomial": {"name": "binomial", "Dist.": {"prior":[("binomial", 1)], "obs":[]}, "D/C": True, "id":["c"]},
"single_regression": {"name": "sgl\_regress", "Dist.": {"prior":[("uniform", 3)], "obs":[("normal", 1)]}, "D/C": False, "id":["b0", "b1", "sigma"]},
"of_blickets_and_blocking": {"name": "blickets", "Dist.": {"prior":[("bernoulli", 9)], "obs":[("bernoulli", 1)]}, "D/C": True},
"the_rectangle_game_with_weak_sampling": {"name": "rect\_game", "Dist.": {"prior":[("uniform", 4)], "obs":[("uniform", 16)]}, "D/C": False, "id":["x1", "x2", "y1", "y2"]},
"lung_cancer": {"name": "lung\_cancer", "Dist.": {"prior":[("bernoulli", 3)], "obs":[]}, "D/C": True, "id":["lungCancer", "cold", "temp"]}, 
"common_cause": {"name": "comm\_cause", "Dist.": {"prior":[("bernoulli", 5)], "obs":[("bernoulli", 1)]}, "D/C": True, "id":["C", "A", "B", "B", "A"]},
"peoples_models_of_coins_expectation": {"name": "ppl\_coins", "Dist.": {"prior":[("uniform", 1),("binomial", 1)], "obs":[("binomial", 1)]}, "D/C": False, "id":["p", "coinSpinner"]},
"polyas_urn": {"name": "polyas\_urn", "Dist.": {"prior":[("bernoulli", 1)], "obs":[]}, "D/C": True},
"true_obs": {"name": "true\_obs", "Dist.": {"prior":[("normal", 2)], "obs":[("normal", 1)]}, "D/C": False, "id":["trueX", "obsX"]},
"posterior_prediction": {"name": "post\_pred", "Dist.": {"prior":[("uniform", 1)], "obs":[("binomial", 2)]}, "D/C": False, "id":["p"]},
"unknown_numbers_of_categories": {"name": "\#categories", "Dist.": {"prior":[("uniform", 2)], "obs":[("uniform", 8), ("binomial", 16)]}, "D/C": False},
"preferences": {"name": "prefer", "Dist.": {"prior":[("uniform", 1), ("bernoulli", 1)], "obs":[("categorical", 9), ("binomial", 3)]}, "D/C": False},
"estimating_causal_power" : {"name": "causal\_pwr", "Dist.": {"prior":[("uniform", 2)], "obs":[("bernoulli", 8)]}, "D/C": False, "id":["cp", "b"]},
"zeroone" : {"name": "zeroone", "Dist.": {"prior":[("uniform", 2)], "obs":[("softmax", 20)]}, "D/C": False, "id":["w1", "w2"]},
"altermu" : {"name": "altermu", "Dist.": {"prior":[("normal", 3)], "obs":[("normal", 40)]}, "D/C": False, "id":["mu[1]", "mu[2]", "mu[3]"]},
"normal_mixture" : {"name": "gauss_mix", "Dist.": {"prior":[("uniform", 1), ("normal", 2)], "obs":[("normal", 80)]}, "D/C": False, "id":["theta", "mu[1]", "mu[2]"]},
"tug" : {"name": "tug", "Dist.": {"prior":[("uniform", 2)], "obs":[("normal", 4), ("bernoulli", 40)]}, "D/C": False, "id":["alice", "bob"]},
"neural" : {"name": "neural", "Dist.": {"prior":[("uniform", 2)], "obs":[("bernoulli-logit", 39)]}, "D/C": False, "id":["w[1]", "w[2]"]},
"timeseries" : {"name": "t-series", "Dist.": {"prior":[("uniform", 3)], "obs":[("normal", 199)]}, "D/C": False, "id":["alpha", "beta", "lag"]}

}

dists_to_params = {
    "bernoulli": {"dist": True, "params": ["p"], "sym": "\mathcal{B}"},
    "flip": {"dist": True, "params": ["p"], "sym": "\mathcal{B}"},
    "binomial": {"dist": True, "params": ["n", "p"], "sym": "\mathcal{B}_m"},
    "uniform": {"dist": False, "params": ["lb.", "ub."], "sym": "\mathcal{U}"},
    "uniformInt": {"dist": True, "params": ["lb.", "ub."], "sym": "\mathcal{U}_I"},
    "gamma": {"dist": False, "params": ["\\alpha", "\\beta"], "sym": "\Gamma"},
    "normal": {"dist": False, "params": ["\\mu", "\\sigma"], "sym": "\mathcal{N}"},
    "gauss": {"dist": False, "params": ["\\mu", "\\sigma"], "sym": "\mathcal{N}"},
    "beta": {"dist": False, "params": ["\\alpha", "\\beta"], "sym": "\\beta"},
    "categorical": {"dist": True, "params": [""], "sym": "mathcal{C}"},
    "softmax": {"dist": False, "params": [""], "sym": "\mathcal{M}"},
    "bernoulli-logit": {"dist": False, "params": ["\\alpha"], "sym": "\mathcal{B}_{log}"},
}

metric_to_str = {
    "KS" : "KS Distance",
    "expdist1" : "Expectation Distance 1"
}

# dist : {"prior":[("bernoulli", 3)], "obs":[("bernoulli", 1)]}
def dist_formatter(dist):
    prior = ""
    for p_tup in dist["prior"]:
        dname, num = p_tup
        if prior == "":
            prior += dists_to_params[dname]["sym"] + "^{" + str(num) + "}"
        else:
            prior += "\\times" + dists_to_params[dname]["sym"] + "^{" + str(num) + "}"
    obs = ""
    for p_tup in dist["obs"]:
        dname, num = p_tup
        obs += "\\times" + dists_to_params[dname]["sym"] + "^{" + str(num) + "}"
    s = "\\underline{{{prior}}}".format(prior=prior)
    if obs != "":
        s += obs
    return "${s}$".format(s=s)

def dict_concator(dicts):
    t = {}
    for d in dicts:
        t.update(d)
    return t

def manuallog2row(log, m, v, p, multirow):
    def all_params(p):
        d2p = dists_to_params
        ps = functools.reduce(operator.iconcat,
                map(lambda d: itertools.product([d2p[d]["sym"]], d2p[d]["params"]),
                    functools.reduce(operator.iconcat, map(lambda tup: [tup[0]] * tup[1], p), []))
                             ,[])
        return ps
    ps = all_params(model_dists[m]["Dist."]["prior"])
    vid = model_dists[m]["id"].index(v)
    param_id = sum([len(dists_to_params[d]["params"]) for d in list(itertools.chain(*[[d] * num for (d,num) in model_dists[m]["Dist."]["prior"]]))[:vid]]) + p
    
    # possible multirow
    nrows = len(model_dists[m]["id"])
    mname = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=model_dists[m]["name"]) if multirow else ""
    d = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=dist_formatter(model_dists[m]["Dist."])) if multirow else ""
    dorc = "D" if model_dists[m]["D/C"] else "C"
    dorc = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=dorc) if multirow else ""
    
    conv = "T"

    
    param = ps[param_id]
    param = "$" + param[0] + ", " + param[1] + "$"

    best_rel_err = "E"
    best_abs_err = "E"
    tsetup = float(search("Setting up GPU, Torch Took: \d+\.\d+", log)[0][28:])
    tt = time_formmater(search("\d+:\d+\.\d+", log)[0]) - tsetup
    tsla = float(search("Trans: \d+\.\d+", log)[0][7:]) / len(model_dists[m]["id"])
    
    sp = "$\infty$"
    if search("EXACT", log):
        ps = float(search("\[\d+.\d+\]", log)[0][1:-1])
        sp = ps / (tsla + tt)
    elif search("PSENSE TO", log):
        ps = "T.O."
    elif search("PSENSE ERR", log):
        ps = "ERR."

    row = {"Model": mname, "Dist.": d, "D/C": dorc, "Param": param,
            "\#splits": "$Sup(M)$", "Conv": conv, "Err\%": best_rel_err, "|Err|": best_abs_err, 
            "PSense": ps, "Total": tsla + tt, "Trans": tsla, "SA": tt, "Speedup": sp}
    
    return row

# parse a .json file into a pd row
def log2row(log, plog, multirow):
    def all_params(p):
        d2p = dists_to_params
        ps = functools.reduce(operator.iconcat,
                map(lambda d: itertools.product([d2p[d]["sym"]], d2p[d]["params"]),
                    functools.reduce(operator.iconcat, map(lambda tup: [tup[0]] * tup[1], p), []))
                             ,[])
        return ps
    ps = all_params(model_dists[log["mname"]]["Dist."]["prior"])
    v, p = log["randvar"], int(log["param"]) - 1
    
    # 1 offset in indexing
    if match(".*\[\d+\]", v):
        v = v[:v.index("[")] + "[" + str(int(v[v.index("[") + 1:-1]) + 1) + "]"

    vid = model_dists[log["mname"]]["id"].index(v)
    param_id = sum([len(dists_to_params[d]["params"]) for d in list(itertools.chain(*[[d] * num for (d,num) in model_dists[log["mname"]]["Dist."]["prior"]]))[:vid]]) + p
    
    # possible multirow
    nrows = len(model_dists[log["mname"]]["id"])
    m = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=model_dists[log["mname"]]["name"]) if multirow else ""
    d = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=dist_formatter(model_dists[log["mname"]]["Dist."])) if multirow else ""
    dorc = "D" if model_dists[log["mname"]]["D/C"] else "C"
    dorc = "\multirow{{{r}}}*{{{t}}}".format(r=nrows, t=dorc) if multirow else ""
    
    spl = int(log["splits"][-1])
    conv = "T" if log["converged"] else "F"

    param = ps[param_id]
    param = "$" + param[0] + ", " + param[1] + "$"

    num_runs = len(list(filter(lambda x: x == "OK", log["status"])))
    tmp = list(filter(lambda x: type(x) == float, log["rel_err"][:num_runs]))
    best_rel_err = min(tmp) * 100 if len(tmp) else "N/A"
    best_abs_err = min(log["abs_err"][:num_runs])
    tt = sum(log["tt"][:num_runs]) - sum(log["tsetup"][:num_runs])
    tsla = log["tsla"] / len(model_dists[log["mname"]]["id"])

    # psense stats
    if plog:
        ps_result = list(filter(lambda p: not search("observe", p) and not str.isspace(p), plog.split("%")))[param_id]
        pst = "N/A"
        spup = "N/A"
        if search("(Invalid syntax)|(PSI program error)", ps_result):
            pst = "Err"
            spup = "$\infty$"
        elif search("time out", ps_result):
            pst = "T.O."
            spup = "$\infty$"
        else:
            pst = float(search("\[\d+.\d+\]", ps_result)[0][1:-1])
            spup = pst / (tsla + tt)
    else:
        pst = "T.O."
        spup = "$\infty$"

    # add to global stats
    global global_spl
    global global_t
    global global_rel_err
    global global_abs_err
    

    t = [sum(log["tt"][:i]) - sum(log["tsetup"][:i]) for i in range(1,len(log["splits"]) + 1)]
    
    # if len(global_t) < len(t):
    #     global_t = np.pad(global_t, [(0, max(len(global_t), len(t)) - len(global_t))])
    # elif len(global_t) > len(t):
    #     t = np.pad(t, [(0, max(len(global_t), len(t)) - len(t))])
    # global_t = global_t + t

    # tmp2 = [x if type(x) == float else 0 for x in log["rel_err"]]
    # if len(global_rel_err) < len(tmp2):
    #     global_rel_err = np.pad(global_rel_err, [(0, max(len(global_rel_err), len(tmp2)) - len(global_rel_err))])
    # elif len(global_rel_err) > len(tmp2):
    #     tmp2 = np.pad(tmp2, [(0, max(len(global_rel_err), len(tmp2)) - len(tmp2))])
    # global_rel_err = global_rel_err + tmp2
    for i in range(len(t)):
        global_t[i].append(t)
        if  type(log["rel_err"][i]) == float:
            global_rel_err[i].append(log["rel_err"][i])
        global_abs_err[i].append(log["abs_err"][i])

    row = {"Model": m, "Dist.": d, "D/C": dorc, "Param": param,
            "\#splits": spl, "Conv": conv, "Err\%": best_rel_err, "|Err|": best_abs_err, 
            "PSense": pst, "Total": tsla + tt, "Trans": tsla, "SA": tt, "Speedup": spup}
    
    return row

# manually examined models
manual = ["coins", 
			"bayes_rule",
			"lung_cancer",
			"murderMystery",
			"common_cause",
			"binomial",
			"explain_away"]

# parse .json and .log files into pd rows
def table_pretty(temps, stans, results):
    
    logs = []

    # load the json log files
    for m in temps + stans:
        d = (temp_dir if m in temps else stan_dir) + m + '/'
        flist = os.listdir(d)
        multirow = True
        if m not in manual:
            for jdir in flist:
                if (jdir.endswith(".json")):
                    randvar = jdir.split("_")[-2]
                    p = jdir.split("_")[-1][:-5]
                    with open(d + jdir, "r") as f:
                        try:
                            with open(d + randvar + "/output", "r") as f2:
                                logs.append((json.load(f), f2.read(), multirow))
                                multirow = False
                        except IOError:
                            logs.append((json.load(f), None, multirow))
                            multirow = False
                
        else:
            for randvar in model_dists[m]["id"]:
                flist = os.listdir(d + randvar)
                for ldir in flist:
                    if (ldir.endswith(".log")):
                        with open(d + randvar + "/" + ldir, "r") as f:
                            results.append(manuallog2row(f.read(), m, randvar, int(ldir.split("_")[-1][:-4]) - 1, multirow))
                            multirow = False
    for log in logs:
        results.append(log2row(*log))

    return results

# '0:05.70' -> float(5.70)
def time_formmater(txt):
    mins = split(':|\.', txt)[0]
    secs = split(':|\.', txt)[1]
    milisecs = split(':|\.', txt)[2]
    return int(mins) * 60 + int(secs) + float(milisecs)/100    

def avg_dist(true_supp, true_value, supp, value, is_ratio=False):
    if (true_supp == supp):
        return sum(abs(true_value - value))/len(supp)
    else:
        # truncate supp, so that supp issubseteq true_supp
        while supp[0] < true_supp[0]:
            supp = supp[1:]
            value = value[1:]
        while supp[-1] > true_supp[-1]:
            supp = supp[:-1]
            value = value[:-1]
        r = 0
        j = 0
        valid = 0
        for i in range(len(supp)):
            while not (true_supp[j] <= supp[i] and supp[i] <= true_supp[j+1]):
                j += 1
            if is_ratio and true_value[j] != 0 and true_value[j] > DIST_THRES:
                r += abs(true_value[j] - value[i]) / true_value[j]
                valid += 1
            elif is_ratio and true_value[j] == 0:
                continue
            elif not is_ratio:
                r += abs(true_value[j] - value[i])
        if not is_ratio:
            return r / len(supp)
        elif is_ratio and valid:
            return r / valid
        else:
            return "N/A"

def plot_converge(mname, randvar, param, epslist, dist, store, splits, rel_err, abs_err, tsetup, tt, plot_dir, converged):
    # make the figure
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)

    mark = ["v","s","o"]
    colors = ["#F0A000","#FF0000", "purple"]

    ax.plot(epslist, dist, c='C0', lw=3.5, label="True ED", zorder=10)

    if (len(store) > 3):
        store = [store[0], store[int(len(store)/2)], store[-1]]
        rel_err = [rel_err[0], rel_err[int(len(rel_err)/2)], rel_err[-1]]
        abs_err = [abs_err[0], abs_err[int(len(abs_err)/2)], abs_err[-1]]
        splits = [splits[0], splits[int(len(splits)/2)], splits[-1]]
        tsetup = [tsetup[0], tsetup[int(len(tsetup)/2)], tsetup[-1]]
        tt = [tt[0], tt[int(len(tt)/2)], tt[-1]]

    i = 1     
    for sense_epslist,sense_dist in store:
        # plot
        alp = i/(len(store)+1)/1.2 + 0.15
        sense_epslist,sense_dist = sense_epslist[0:len(sense_epslist):4],sense_dist[0:len(sense_dist):4]
        ax.plot(sense_epslist, sense_dist, linewidth=0, markersize=10,
            marker=mark[i-1], markerfacecolor='none', markeredgecolor=colors[i-1],
            markeredgewidth=2.5, label="#splits={s}".format(s=splits[i-1]), zorder=11)		
        
        # if i == len(store):
        #     rel = "%.2f" %rel_err[-1] if type(rel_err[-1]) == float else rel_err[-1]
        #     abs = "%.4f" %abs_err[-1] if type(abs_err[-1]) == float else abs_err[-1]
        #     text = "rel_err={r}\nabs_err={a}".format(r = rel, a = abs)
        #     text2 = "converged={c}\n".format(c=converged)
        #     ax.annotate(text2 + text, # this is the text
        #             (sense_epslist[-1], sense_KS[-1]), # these are the coordinates to position the label
        #             textcoords="offset points", # how to position the text
        #             xytext=(5,5), # distance from text to points (x,y)
        #             ha='center', # horizontal alignment can be left, right or center
        #             fontsize=8) 
        i += 1

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel("eps", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc="upper left", fontsize=20)

    fig.savefig(plot_dir, format="pdf")

def log_rq1(log_dir, mname, randvar, param, tsla, splits, tsetup, tt, status, rel_err, abs_err, converged):
    results = {}
    results["mname"] = mname
    results["randvar"] = randvar
    results["param"] = param
    results["tsla"] = tsla
    results["splits"] = splits
    results["tsetup"] = tsetup
    results["tt"] = tt
    results["status"] = status
    results["rel_err"] = rel_err
    results["abs_err"] = abs_err
    results["converged"] = converged
    return results

def compare_dist(mname, metric, truth_dir, isStan, randvar, param, epslist, range, dist, matrix, rq, plot_dir, log_dir, is_converged, spl_st, max_to, tsla):
    lower = epslist[0]
    upper = epslist[-1]

    # check if sense script exists, if not, run translator
    dir_script = "./benchmarks/" + ("stan_bench/" if isStan else "psense_bench/") + mname + "/" + mname + ".py"
    
    # run sense script with different #splits (-s) until convergence to ground truth
    s = spl_st

    # Err: distance btw AquaSense's sensitivity and true sensitivity
    splits, rel_err, abs_err, tsetup, tt, status = [],[],[],[],[],[]

    store = []
    converged = False
    while(True):
        script = ["/usr/bin/time", "python3", dir_script, "-d", metric, "-n", "40", "-s", str(s), "-v", randvar, "-p", str(param - 1), "-b", str(lower), str(upper)]
        script = " ".join(script)
        stdout, stderr, TERM = Command(script, timeout=max_to).run(capture=True)

        # check for errors, ERR / OOM / T.O
        if (TERM):
            status.append("T.O.")
            break
        if search("CUDA out of memory", stderr):
            status.append("OOM.")
            break
        if search("exited with non-zero status", stderr):
            status.append("Err.")
            break
        status.append("OK")

        # record time taken for setup, and AquaSense total
        tsetup.append(float(search("\d+.\d+", stdout)[0]))
        tt.append(time_formmater(findall("\d+:\d+\.\d+", stderr)[0]))

        sense_dir = "./benchmarks/" + ("stan_bench/" if isStan else "psense_bench/") + mname + "/" + "_".join(["sense", mname, randvar, str(param), metric]) + ".csv"
        with open(sense_dir) as c:
            text = c.read()
            sense_epslist = np.fromstring(text.split('\n')[0], sep=',')
            sense_dist = np.fromstring(text.split('\n')[1], sep=',')
            store.append((sense_epslist, sense_dist))
            
            # compute the AVERAGE distance between ground truth sensitivity and AquaSense sensitivity
            abs_err.append(avg_dist(*[x.tolist() for x in [epslist, dist, sense_epslist, sense_dist]], is_ratio=False))
            rel_err.append(avg_dist(*[x.tolist() for x in [epslist, dist, sense_epslist, sense_dist]], is_ratio=True))
            splits.append(s)
        
        # check convergence
        c1 = is_converged(rel_err, s, REL_THRES)
        c2 = is_converged(abs_err, s, ABS_THRES)
        if c1 == 0 or c2 == 0:
            converged = True
            break
        elif c1 == 1 and c2 == 1:
            break
        s *= 2
    
    if (plot_dir):
        plot_converge(mname, randvar, param, epslist, dist, store, splits, rel_err, abs_err, tsetup, tt, plot_dir, converged)
    
    if (log_dir):
        j = log_rq1(log_dir, mname, randvar, param, tsla, splits, tsetup, tt, status, rel_err, abs_err, converged)
        with open(log_dir, "w") as f:
            json.dump(j, f)
    
# 1 is fail to converge, 0 is converged, -1 is undertermined (keep trying)
def is_converged(err, s, thres, max_splits = 16000, rate = 0.95):
    if (type(err[-1]) == float and err[-1] <= thres):
        return 0
    # if (len(err) > 1 and err[-1] > rate * err[-2]):
    #     return 1
    if s > max_splits:
        return 1
    return -1

# parse ground truth and run AquaSense
def model(metric, isStan, mname, RQ):
    # obtain the ground truth of the model
    bench = "stan_bench/" if isStan else "psense_bench/"
    flist = os.listdir("./benchmarks/" + bench + mname)
    tempsrc = os.path.dirname(os.path.realpath(__file__)) + "/benchmarks/" + bench + mname + "/" + "" if isStan else list(filter(lambda x: x.endswith(".template"), flist))[0]
    
    csvlist = []
    randvars = []
    params = []
    for fname in flist:
        if (fname.startswith(mname) and fname.endswith("{m}.csv".format(m=metric))):
            csvlist.append(fname)
            randvars.append(fname[len(mname)+1:-4].split("_")[0])
            params.append(fname[len(mname)+1:-4].split("_")[1])
    
    # check if there is no ground truth, needs to run mathematica script / psense
    if (randvars == []):
        logging.warning("WARNING: no ground truth exist for model ", mname, ", skipped (run mathematica script / psense)")
        return

    # translate model to .py
    translator = ["/usr/bin/time", java_bin, "-cp", "/home/zitongzhou/sense/target/aqua-1.0.jar:/home/zitongzhou/sense/lib/storm-1.0.jar", "aqua.analyses.PyCompilerRunner", tempsrc]
    
    proc = subprocess.run(translator, capture_output=True)
    exetime_str = proc.stderr.decode("utf-8")
    tsla = time_formmater(findall("\d+:\d+\.\d+", exetime_str)[0])
    
    # for each randvar/param, parse its ground truth and call compare
    for i in range(len(csvlist)):
        truth_dir = "./benchmarks/" + ("stan_bench/" if isStan else "psense_bench/") + mname + "/" + csvlist[i]
        
        with open (truth_dir) as c:
            text = c.read()
            epslist = np.fromstring(text.split('\n')[0], sep=',')
            rrange = np.fromstring(text.split('\n')[1], sep=',')
            dist = np.fromstring(text.split('\n')[3], sep=',')

            # parse matrix
            # mtext = text.split('\n')[2][1:-1].replace("{", "").replace("}","").replace("\",\"",";").replace("*^", "e")
            # matrix = np.matrix(mtext)

            # call compare
            if RQ == 1:
                log_dir = "./benchmarks/" + ("stan_bench/" if isStan else "psense_bench/") + mname + "/" + \
                    "_".join(["bench", mname, randvars[i], params[i]]) + ".json"
                plot_dir = "./results/RQ1/fig/" + "_".join(["bench", mname, randvars[i], params[i]]) + ".pdf"
                compare_dist(mname, metric, truth_dir, isStan, randvars[i], int(params[i]), epslist, rrange, dist,\
                    None, 1, plot_dir, log_dir, partial(is_converged, max_splits=100000, rate=0.95), 100, 1200, tsla)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="AquaSense benchmark script", description='')

    parser.add_argument("-rq1", action="store_true", help="Use this option run RQ1")
    parser.add_argument("-table", action="store_true", help="Use this option print latex table")
    parser.add_argument("-m", choices=["KS", "expdist1"], help="Specify distance metric")

    args = parser.parse_args()
    metric = args.m
    # read benchmark.json
    with open('./sense_benchmark_list.json') as f:
        j = json.load(f)
        
        if args.rq1:
            # RQ1, compare Sense's capability/time with PSense
            
            template_models = j["RQ1"]["template"]
            for m in template_models:
                if m not in manual:
                    model(metric, isStan = False, mname = m, RQ = 1)
            # stan models
            stan_models = j["RQ1"]["stan"]
            for m in stan_models:
                if m not in manual:
                    model(metric, isStan = True, mname = m, RQ = 1)
            
        if args.table:
            results = []
            table_pretty(j["RQ1"]["template"], j["RQ1"]["stan"], results)
            table = pd.DataFrame(results)

            # stylish = {"PSense": "%.2f", "Total(s)": ":,.2f", "Trans(s)": ":,.2f", "S.A.(s)": ":,.2f"}
            with open("./results/RQ1/{m}/RQ1Table1.tex".format(m=metric), "w") as f:
                f.write(table.style.hide(axis="index").format({}, precision=2).to_latex())

        
