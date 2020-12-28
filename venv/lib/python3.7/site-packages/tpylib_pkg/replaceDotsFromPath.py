#!/usr/bin/env python

# import sys
import argparse
# import subprocess
import os
# import matplotlib.pyplot as plt

# from tpylib_statistics import TimeseriesStats
from tpylib_FileFolderUtils import FileFolderUtils
# from tpylib_ulgParser import UlgParser
# from tpylib_ulgPlotter import UlgPlotter


class Process:
    def __init__(self, bdir):
        self.bdir = bdir

    def process_file(self, pfile):
        dirname = os.path.dirname(pfile)
        basename = FileFolderUtils.get_basename(pfile)
        print('[process_file] processing %s' % pfile)
        assert isinstance(basename, str)
        nbasename = basename.replace('0.0', '0p0')
        npfile = FileFolderUtils.full_path(dirname + '/' + nbasename)

        print('[process_file] new filename %s' % npfile)
        os.rename(pfile, npfile)

    def process_bdir(self):
        print('[process_bdir] processing %s' % self.bdir)

        # foldername, mextension, method
        FileFolderUtils.run_method_on_folder(self.bdir, '',
                                             ulgprocess.process_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do stuff')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory')
    # parser.add_argument('--plot', action='store_true', required=False,
    #                     help='plot results')
    # parser.add_argument('--loc', action='store_true', help='location')
    args = parser.parse_args()

    bdir = FileFolderUtils.full_path(args.bdir)
    # uplot = args.plot

    foldername_arr = FileFolderUtils.get_folder_arr(bdir, 'Angle')
    print(foldername_arr)
    for folder in foldername_arr:
        ulgprocess = Process(bdir)
        ulgprocess.process_bdir()
        assert isinstance(bdir, str)
        newname = bdir.replace('0.0', '0p0')
        os.rename(bdir, newname)
