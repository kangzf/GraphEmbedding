

class Data(object):
    def __init__(self):
        print('Data ctor')

    def name(self):
        return self.__class__.__name__


class SynData(Data):
    def __init__(self, train):
        # super().__init__()
        print('Syndata ctor train %s ' % train)

