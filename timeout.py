# https://raw.githubusercontent.com/bo-yang/misc/master/run_command_timeout.py

import subprocess
import threading

""" Run system commands with timeout
"""
class Command(object):
    def __init__(self, cmd, timeout):
        self.cmd = cmd
        self.process = None
        self.out = None
        self.err = None
        self.timeout = timeout
        self.TERM = False

    def run_command(self, capture = False):
        if not capture:
            self.process = subprocess.Popen(self.cmd,shell=True)
            self.process.communicate()
            return
        # capturing the outputs of shell commands
        self.process = subprocess.Popen(self.cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        out, err = self.process.communicate()
        self.err = err.decode("utf-8")
        if len(out) > 0:
            self.out = out.decode("utf-8")
        else:
            self.out = None

    # set default timeout to 2 minutes
    def run(self, capture = False):
        thread = threading.Thread(target=self.run_command, args=(capture,))
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            # print('Command TIMEOUT, KILLED: ' + self.cmd)
            self.process.terminate()
            self.TERM = True
            thread.join()
        return self.out, self.err, self.TERM

# '''basic test cases'''

# # run shell command without capture
# t = time.time()
# Command('python3 ~/PSense/psense.py -f RQ1/gamma.psi', timeout=10).run()
# print(time.time() - t)

# script = ["/usr/bin/time", "python3", "/home/zitongzhou/sense/benchmarks/stan_bench/normal_mixture/normal_mixture.py", "-d", "KS", "-s", "2000"]
# script = " ".join(script)
# stdout, stderr, TERM = Command(script, timeout=2000).run(capture=True)

# i = 1