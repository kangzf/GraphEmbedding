import pickle
from utils import get_root_path


class Data(object):
    def __init__(self, train):
        name = self.__class__.__name__ + '_'
        self.train = True if train else False
        if train:
            name += 'train'
        else:
            name += 'test'
        self.name = name
        sfn = self.save_filename()
        try:
            self.load()
            print('%s loaded from %s' % (name, sfn))
        except:
            self.init()
            self.save()
            print('%s saved to %s' % (name, sfn))

    def save(self):
        file = open(self.save_filename(), 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        file = open(self.save_filename(), 'rb')
        dp = file.read()
        file.close()
        self.__dict__ = pickle.loads(dp)

    def save_filename(self):
        return '{}/save/{}.txt'.format(get_root_path(), self.name)


class SynData(Data):
    def __init__(self, train):
        super().__init__(train)

    def init(self):
        print('actually init')


def create_syn_data():
    pass
