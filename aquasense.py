#!/usr/bin/python3

# aquasense.py

import sys, os, logging, subprocess, argparse
import numpy as np
import matplotlib.pyplot as plt
from timeout import Command
from re import search, findall, split
import converge

java_bin = "java"
logging.getLogger().setLevel(logging.INFO)

ERR_CODE = -256
TO_CODE = -128

# convergence criteria
DIST_THRES = 1e-6
REL_THRES = 0.05
ABS_THRES = 1e-6

# Minimum #splits to start doubling from
# Adjust it according to memory and execution time cost
SPL_MIN = 100

# Maximum time-out for a single run of AquaSense
MAX_TO = 1200

metric_to_str = {
    "KS" : "KS Distance",
    "expdist1" : "Expectation Distance 1"
}

# '0:05.70' -> float(5.70)
def time_formmater(txt):
    mins = split(':|\.', txt)[0]
    secs = split(':|\.', txt)[1]
    milisecs = split(':|\.', txt)[2]
    return int(mins) * 60 + int(secs) + float(milisecs)/100    

def plot_converge(mname, randvar, param, store, splits, tsetup, tt, converged):
    # make the figure
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)

    mark = ["v","s","o"]
    colors = ["#F0A000","#FF0000", "purple"]

    if (len(store) > 3):
        store = [store[0], store[int(len(store)/2)], store[-1]]
        # rel_err = [rel_err[0], rel_err[int(len(rel_err)/2)], rel_err[-1]]
        # abs_err = [abs_err[0], abs_err[int(len(abs_err)/2)], abs_err[-1]]
        splits = [splits[0], splits[int(len(splits)/2)], splits[-1]]
        tsetup = [tsetup[0], tsetup[int(len(tsetup)/2)], tsetup[-1]]
        tt = [tt[0], tt[int(len(tt)/2)], tt[-1]]

    i = 1     
    for sense_epslist,sense_dist in store:
        # plot
        sense_epslist,sense_dist = sense_epslist[0:len(sense_epslist):4],sense_dist[0:len(sense_dist):4]
        ax.plot(sense_epslist, sense_dist, linewidth=0, markersize=10,
            marker=mark[i-1], markerfacecolor='none', markeredgecolor=colors[i-1],
            markeredgewidth=2.5, label="#splits={s}".format(s=splits[i-1]), zorder=11)		
        
        i += 1

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel("eps", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc="upper left", fontsize=20)

    plt.show()

def compare_dist(mname, metric, isStan, randvar, param, bounds, is_converged, spl_st, max_to, ttsla):
    lower = bounds[0] if bounds else None
    upper = bounds[-1] if bounds else None

    dir_script = "./benchmarks/" + ("stan_bench/" if isStan else "psense_bench/") + mname + "/" + mname + ".py"
    
    # run sense script with different #splits (-s) until convergence to ground truth
    s = spl_st

    splits, tsetup, tt, status = [],[],[],[]

    store = []
    converged = False
    while(not converged):
        if randvar:
            script = ["/usr/bin/time", "python3", dir_script, "-d", metric, "-n", "40", "-s", str(s), "-v", randvar, "-p", str(param), "-b", str(lower), str(upper)]
        else:
            script = ["/usr/bin/time", "python3", dir_script, "-d", metric, "-n", "40", "-s", str(s)]
        script = " ".join(script)
        stdout, stderr, TERM = Command(script, timeout=max_to).run(capture=True)

        # check for errors, ERR / OOM / T.O
        if (TERM):
            status.append("T.O.")
            if len(store):
                logging.warning("Timed-out at #splits=" + str(s))
            else:
                logging.error("CUDA timed-out at MAX_TO=" + str(max_to), ", SPL_MIN=" + str(s) + ". Try increase MAX_TO or reduce SPL_MIN.")    
            break
        if search("CUDA out of memory", stderr):
            status.append("OOM.")
            if len(store):
                logging.warning("CUDA out of memory at #splits=" + str(s))
            else:
                logging.error("CUDA out of memory at SPL_MIN=" + str(s) + ". Try reduce SPL_MIN.")
            break
        if search("exited with non-zero status", stderr):
            status.append("Err.")
            logging.error("Internal error at #splits=" + str(s))
            break
        status.append("OK")

        # record time taken for setup, and AquaSense total
        tsetup.append(float(search("\d+.\d+", stdout)[0]))
        tt.append(time_formmater(findall("\d+:\d+\.\d+", stderr)[0]))

        # parse sensitivity
        sense_dir = search("Saved to .*", stdout)[0][10:]
        
        with open(sense_dir) as c:
            text = c.read()
            sense_epslist = np.fromstring(text.split('\n')[0], sep=',')
            sense_dist = np.fromstring(text.split('\n')[1], sep=',')
            store.append((sense_epslist, sense_dist))
            
            splits.append(s)
        
        # check convergence
        converged = is_converged(splits, store)
        s *= 2
    
    if len(store):
        logging.info("Most accurate sensitivity data is stored at: " + sense_dir)

    plot_converge(mname, randvar, param, store, splits, tsetup, tt, converged)
    
# run AquaSense on all parameters
def model(metric, isStan, path, mname, randvar, param, bounds):

    # translate model to .py
    bench = "stan_bench/" if isStan else "psense_bench/"
    # flist = os.listdir("./benchmarks/" + bench + mname)
    dirname = os.path.dirname(os.path.realpath(__file__))
    tempsrc = dirname + "/" + path + "/" + ((mname + ".template" ) if not isStan else "")
    
    translator = ["/usr/bin/time", java_bin, "-cp", "{dir}/target/aqua-1.0.jar:{dir}/lib/storm-1.0.jar".format(dir=dirname), "aqua.analyses.PyCompilerRunner", tempsrc]
    
    # capture_output is added to subprocess in Python 3.7+
    # proc = subprocess.run(translator, capture_output=True)
    proc = subprocess.run(translator, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    exetime_str = proc.stderr.decode("utf-8")
    tsla = time_formmater(findall("\d+:\d+\.\d+", exetime_str)[0])
    
    # call compare
    compare_dist(mname, metric, isStan, randvar, param, bounds, converge.is_converged, SPL_MIN, MAX_TO, tsla)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="AquaSense benchmark script", description='')

    parser.add_argument('path', type=str, nargs=1,
                    help='the path to the model')

    # specify which variable to add noise to
    # if no arg, use the first randvar
    parser.add_argument("-v", "--randvar", help="Use this option to specify the random variable to add noise")

    # specify which parameter to add noise to
    # if no arg, use the first parameter
    parser.add_argument("-p", "--parameter", nargs=1, type=int, help="Use this option to specify the parameter index to add noise")

    parser.add_argument("-b", "--bounds", nargs=2, type=float, help="Use this option to specify the lower & upper bound of eps, e.g. -v beta -p 0 -b -0.5 0.5")

    parser.add_argument("-s", "--splits", nargs=1, type=int, help="Use this option to specify the starting #splits to double from")
   
    parser.add_argument("-m", help="Specify distance metric, to use a customized metric, add it to metrics.py")
    
    parser.add_argument("-c", help="Use a customized convergence, add it to metrics.py")

    args = parser.parse_args()
    metric = "expdist1"
    if args.m != None:
        metric = args.m
    path = args.path[0].rstrip("/")
    randvar = args.randvar
    param = 0
    if args.parameter:
        param = args.parameter[0]
    if args.bounds != None:
        if (args.bounds[0] >= args.bounds[1]):
            sys.exit("Error: invalid noise bound")
    bounds = args.bounds
    isStan = search("stan_bench", path)

    mname = path.split("/")[-1] if isStan else path.split("/")[-1] 
    
    model(metric, isStan, path, mname, randvar, param, bounds)

        
