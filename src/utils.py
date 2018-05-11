def get_model_fun(model, train):
    import sys
    sys.path.insert(0, get_root_path())
    if model == 'iwge':
        if train:
            from IWGE.src.train import train
            return train
        else:
            from IWGE.src.test import test
            return test
    else:
        raise RuntimeError('Not recognized model %s' % model)


def load_data(data, train):
    if data == 'syn':
        from data import SynData
        return SynData(train)
    elif data == 'aids10k':
        from data import AIDS10kData
        return AIDS10kData(train)
    else:
        raise RuntimeError('Not recognized data %s' % data)


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def draw_graph(g, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure()
    import networkx as nx
    nx.draw(g, ax=f.add_subplot(111))
    f.savefig(file)
    print('Saved graph to {}'.format(file))


def exec(cmd, timeout=None):
    if not timeout:
        from os import system
        print(cmd)
        system(cmd)
    else:
        import subprocess as sub
        import threading

        class RunCmd(threading.Thread):
            def __init__(self, cmd, timeout):
                threading.Thread.__init__(self)
                self.cmd = cmd
                self.timeout = timeout

            def run(self):
                self.p = sub.Popen(self.cmd, shell=True)
                self.p.wait()

            def Run(self):
                self.start()
                self.join(self.timeout)

                if self.is_alive():
                    self.p.terminate()
                    self.join()
                    self.finished = False
                else:
                    self.finished = True

        print('Timed cmd {}sec {}'.format(timeout, cmd))
        r = RunCmd(cmd, timeout)
        r.Run()
        return r.finished


tstamp = None


def get_ts():
    import datetime
    global tstamp
    if not tstamp:
        tstamp = datetime.datetime.now().isoformat()
    return tstamp

def get_file_base_id(file):
    return int(file.split('/')[-1].split('.')[0])
