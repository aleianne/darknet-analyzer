from pathlib import Path

from pair_object import PairObject
from utils.exceptions import ObjectTypeException


class CfgFileHandler:

    def __init__(self, file_pointer, filename):
        self.file_ptr = file_pointer
        self.cfg_object = CfgFileObject(filename)

    def load_config(self):

        for line in self.file_ptr:

            if line[0] == "\n":
                # the line is empty
                pass
            elif line[0] == '#':
                # if the line begin with #
                pass
            elif line[0] == '[':
                self.cfg_object.add_section(self._strip_section_name(line))
            else:
                self.cfg_object.add_new_option(line)

    def get_configuration_object(self):
        return self.cfg_object

    def _strip_section_name(self, section):

        c = section[0]
        if c != "[":
            raise Exception()

        c = section[-2]
        if c != "]":
            raise Exception()

        c = section[-1]
        if c != "\n":
            raise Exception()

        # section name split from the []\n char
        s_section = section.strip("[]\n")

        return s_section


class CfgFileObject:

    def __init__(self, name):
        self.name = name
        self.sections = []
        self.n = 0          # number of section into the cfg object

    def add_section(self, section_name):
        new_section = CfgFileSection(section_name)
        self.sections.append(new_section)
        self.n += 1

    def add_new_option(self, line):

        # first strip the line from \n and than split by = value
        line = line.strip("\n")
        s = line.split('=')

        if len(s) == 0:
            raise Exception

        key = s[0]
        value = s[1]

        new_section = self.sections[self.n - 1]
        new_section.add_opt(key, value)

    def del_section(self, n):
        self.sections.remove(n)

    def get_section(self, n):
        return self.sections[n]

    def get_all_sections(self):
        return self.sections

    def get_section_number(self):
        return self.n

    def get_file_name(self):
        return self.name


class CfgFileSection:

    def __init__(self, section_name):
        self.section_name = section_name
        self.options = []

    def add_opt(self, key, value):
        new_pair = PairObject(key, value)
        self.options.append(new_pair)

    def del_opt(self, n):
        self.options.remove(n)

    def find_opt(self, key):
        for pair in self.options:
            if pair.get_key() == key:
                return pair.get_value()

        return None

    def update_opt(self, n, key=None, value=None):
        pair = self.options[n]

        if key is not None:
            pair.set_key(key)

        if value is not None:
            pair.set_value(value)

    def get_section_name(self):
        return self.section_name

    def print_all_options(self):
        print("the section name is", self.section_name)
        for i in self.options:
            print(i.string_pair())
        print("\n")

    def _check_list_bound(self, n):
        pass


def load_configuration_file(file):

    if not isinstance(file, Path):
        if isinstance(file, str):
            file = Path(file)
        else:
            raise ObjectTypeException(file)

    if not file.exists():
        print("the configuration file", file.as_posix(), "doesn't exists")
        raise FileNotFoundError

    cfg_handler = None

    with open(file, 'r') as fp:
        # create a new Cfg File Object
        cfg_handler = CfgFileHandler(fp, file.as_posix())
        cfg_handler.load_config()

    configuration_file = cfg_handler.get_configuration_object()
    return configuration_file
