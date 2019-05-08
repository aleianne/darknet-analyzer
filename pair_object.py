class PairObject:

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def set_key(self, key):
        self.key = key

    def set_value(self, value):
        self.value = value

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def string_pair(self):
        pair = self.key + " = " + self.value
        return pair
