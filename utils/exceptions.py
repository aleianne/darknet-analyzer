class LayerTypeException(Exception):

    def __init__(self):
        pass


class ObjectTypeException(Exception):

    def __init__(self, object):
        self.obj = object
        self.err_msg = "the object type is "

    def print_err_msg(self):
        print(self.err_msg, end=" ")
        print(type(self.obj))
