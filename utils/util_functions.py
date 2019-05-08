import struct

from pathlib import Path


def generate_filename(filename):
    ''' pass as argument the filename relative to the user home dir '''
    home_path = Path.home()
    filename_path = home_path / filename
    return filename_path


def decode_bytes_from_file(file, size, output_list, d=False):

    # TODO fare attenzione all'allineamento e dell'ordine in cui vengono letti i byte
    bytes_to_float_parser = struct.Struct("f")

    for i in range(0, size):
        read_bytes = file.read(4)

        if d:
            print(read_bytes)

        if len(read_bytes) < 4:
            print("errore gravissimissimo ahahhaha")
            print("l'indice è {i} mentre la dimesione è {d}".format(i=i, d=size))

        # return the floating point representation
        b = bytes_to_float_parser.unpack(read_bytes)
        output_list.append(b[0])
