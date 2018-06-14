def get_model_fun(model, train):
    import sys
    sys.path.insert(0, get_root_path() + '/models')
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
    elif data == 'aids50':
        from data import AIDS50Data
        return AIDS50Data(train)
    elif data == 'aids10k_small':
        from data import AIDS10kSmallData
        return AIDS10kSmallData(train)
    elif data == 'aids10k':
        from data import AIDS10kData
        return AIDS10kData(train)
    else:
        raise RuntimeError('Not recognized data %s' % data)


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return get_root_path() + '/data'


def get_save_path():
    return get_root_path() + '/save'


def get_src_path():
    return get_root_path() + '/src'


def get_model_path():
    return get_root_path() + '/model'


def get_result_path():
    return get_root_path() + '/result'


def draw_graph(g, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure()
    import networkx as nx
    nx.draw(g, ax=f.add_subplot(111))
    f.savefig(file)
    print('Saved graph to {}'.format(file))


exec_print = True


def exec_turnoff_print():
    global exec_print
    exec_print = False


def exec_turnon_print():
    global exec_print
    exec_print = True


def exec(cmd, timeout=None):
    global exec_print
    if not timeout:
        from os import system
        if exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
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

        if exec_print:
            print('Timed cmd {}sec {}'.format(timeout, cmd))
        r = RunCmd(cmd, timeout)
        r.Run()
        return r.finished


tstamp = None


def get_ts():
    import datetime, pytz
    global tstamp
    if not tstamp:
        tstamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%dT%H:%M:%S')
    return tstamp


def get_file_base_id(file):
    return int(file.split('/')[-1].split('.')[0])


def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


def save_as_dict(filepath, *args, **kwargs):
    '''
    Warn: To use this function, make sure to call it in ONE line, e.g.
    save_as_dict('some_path', some_object, another_object)
    Moreover, comma (',') is not allowed in the filepath.
    '''
    import inspect
    from collections import OrderedDict
    frames = inspect.getouterframes(inspect.currentframe())
    frame = frames[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    dict_to_save = OrderedDict()
    all_args_strs = string[string.find('(') + 1:-1].split(',')
    if 1 + len(args) + len(kwargs) != len(all_args_strs):
        msgs = ['Did you call this function in one line?', \
                'Did the arguments have comma "," in the middle?']
        raise RuntimeError('\n'.join(msgs))
    for i, name in enumerate(all_args_strs[1:]):
        if name.find('=') != -1:
            name = name.split('=')[1]
        name = name.strip()
        if i >= 0 and i < len(args):
            dict_to_save[name] = args[i]
        else:
            break
    dict_to_save.update(kwargs)
    print('Saving a dictionary as pickle to {}'.format(filepath))
    save(filepath, dict_to_save)


def load_as_dict(filepath):
    print('Loading a dictionary as pickle from {}'.format(filepath))
    return load(filepath)


def load_pkl(handle):
    import pickle
    return pickle.load(handle)


def save_pkl(obj, handle):
    import pickle
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save(filepath, obj):
    with open(proc_filepath(filepath), 'wb') as handle:
        save_pkl(obj, handle)


def load(filepath):
    from os.path import isfile
    filepath = proc_filepath(filepath)
    if isfile(filepath):
        with open(filepath, 'rb') as handle:
            return load_pkl(handle)
    else:
        return None


def proc_filepath(filepath):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.pickle'
    if ext not in filepath:
        filepath += ext
    return filepath


def prompt(str, options=None):
    while True:
        t = input(str)
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_cpu():
    from os import cpu_count
    while True:
        num_cpu = prompt( \
            '{} cpus available. How many do you want?'.format( \
                cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None


computer_name = None


def get_computer_name():
    from os.path import isfile
    global computer_name
    if not computer_name:
        fp = get_src_path() + '/computer_name.txt'
        if not isfile(fp):
            raise RuntimeError('No {} exists!'.format(fp))
        with open(fp) as f:
            computer_name = f.read().rstrip()
            if '\n' in computer_name:
                raise RuntimeError('Only one line in the computer name {}'. \
                                   format(computer_name))
            if not computer_name:
                raise RuntimeError('Computer name "{}" cannot be empty'. \
                                   format(computer_name))
    return computer_name


def check_nx_version():
    import networkx as nx
    nxvg = '1.10'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError( \
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))
