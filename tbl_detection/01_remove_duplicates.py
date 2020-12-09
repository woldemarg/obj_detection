import sys
import getopt
import hashlib
import os
import itertools
import shutil


def make_md5_hash(file_path):
    md5_hash = hashlib.md5()
    a_file = open(file_path, "rb")
    content = a_file.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()
    return digest


def main(argv):
    try:
        opts = getopt.getopt(argv, "hd:")[0]

    except getopt.GetoptError:
        print(r'usage: remove_duplicates.py -d <path\to\pdf\dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(r'usage: remove_duplicates.py -d <path\to\pdf\dir>')
            sys.exit()

        elif opt == '-d':
            pdf_examples_dir = arg

    hash_dict = {}
    for file in os.listdir(pdf_examples_dir):
        file_name = os.fsdecode(file)
        if file_name.endswith('.jpg'):
            path = os.path.join(pdf_examples_dir, file)
            hash_dict[file] = make_md5_hash(path)

    flip_dict = {}
    for key, value in hash_dict.items():
        if value not in flip_dict:
            flip_dict[value] = [key]
        else:
            flip_dict[value].append(key)

    duplicates_list = list(itertools
                           .chain
                           .from_iterable(
                               [v[1:] for v in flip_dict.values()
                                if len(v) > 1]))

    duplicates_dir = os.path.join(pdf_examples_dir, 'duplicates')
    if not os.path.exists(duplicates_dir):
        os.mkdir(duplicates_dir)

    for f_name in duplicates_list:
        f_path = os.path.join(pdf_examples_dir, f_name)
        shutil.copy(f_path, duplicates_dir)
        os.remove(f_path)


if __name__ == '__main__':
    main(sys.argv[1:])
