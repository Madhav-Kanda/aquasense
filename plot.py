from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shutil import copyfile, rmtree
from re import T, search, findall, split, match

def plot_converge(mname, randvar, param, epslist, dist, store, splits, rel_err, abs_err, tsetup, tt, plot_dir, converged):
    # make the figure
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)

    ax.plot(epslist, dist, c='C0', lw=3.5, label="True ED", zorder=10)
    while(len(store) > 3 and rel_err[0] > 0.2):
        store = store[1:]
        rel_err = rel_err[1:]
        abs_err = abs_err[1:]
        splits = splits[1:]
        tsetup = tsetup[1:]
        tt = tt[1:]

    if (len(store) > 3):
        store = [store[0], store[int(len(store)/2)], store[-1]]
        rel_err = [rel_err[0], rel_err[int(len(rel_err)/2)], rel_err[-1]]
        abs_err = [abs_err[0], abs_err[int(len(abs_err)/2)], abs_err[-1]]
        splits = [splits[0], splits[int(len(splits)/2)], splits[-1]]
        tsetup = [tsetup[0], tsetup[int(len(tsetup)/2)], tsetup[-1]]
        tt = [tt[0], tt[int(len(tt)/2)], tt[-1]]
        
    i = 1
    
    for sense_epslist,sense_KS in store:
        # plot
        alp = i/(len(store)+1)/1.2 + 0.15
        mark = ["v","s","o"]
        colors = ["#F0A000","#FF0000", "purple"]
        ax.plot(sense_epslist, sense_KS, linewidth=0, markersize=10,
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

    title = "Model: {m}, var: {p}, param: {i}".format(m=mname, p=randvar, i=param)
    # ax.set_title(title, fontsize=26, verticalalignment='bottom')
    ax.set_xlabel("eps", fontsize=22)
    ax.set_ylabel("ED", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc="upper left", fontsize=20)

    # Add a table at the bottom of the axes
    # cell_text = []
    # for i in range(len(splits)):
    #     cell_text.append([tt[i] - tsetup[i], rel_err[i], abs_err[i]])
    # the_table = plt.table(cellText=cell_text,
    #                   rowLabels=["#splits=%d" % x for x in splits],
    #                   colLabels=["time", "rel_err", "abs_err"],
    #                   loc='bottom')
    # # Adjust layout to make room for the table:
    # plt.subplots_adjust(left=0.2, bottom=0.1)

    # frame around figure
    # fig.patch.set(linewidth=2, edgecolor='0.5')
    plt.show()
    fig.savefig("./t1.pdf", format="pdf")

# mname = "gamma"
# randvar = "c"
# param = 1
# epslist = np.arange(0, 1, step=0.01)
# dist = np.arange(0, 1, step=0.01)
# sense_epslist = np.arange(0, 1, step=(1-0)/10)
# store = [(sense_epslist, np.arange(0, 0.5, step=(0.5-0)/10)), 
#             (sense_epslist, np.arange(0, 0.7, step=(0.7-0)/10)), 
#             (sense_epslist, np.arange(0, 0.95, step=(0.95-0)/10))]
# splits = [100,200,400]
# rel_err = [0.2, 0.1, 0.04]
# abs_err = [0.002, 0.001, 0.0004]
# tsetup = [2,2,2]
# tt = [3,4.123124234,16.1024134134]
# converged = True
# plot_converge(mname, randvar, param, epslist, dist, store, splits, rel_err, abs_err, tsetup, tt, None, converged)

