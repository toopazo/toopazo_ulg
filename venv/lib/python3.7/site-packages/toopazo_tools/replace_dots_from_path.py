#!/usr/bin/env python

from toopazo_tools.file_folder import FileFolderTools

# import sys
import argparse
# import subprocess
import os
# import matplotlib.pyplot as plt


class Process:
    def __init__(self, bdir):
        self.bdir = bdir

    def process_file(self, pfile):
        dirname = os.path.dirname(pfile)
        basename = FileFolderTools.get_file_basename(pfile)
        print('[%s] processing %s' % (self.process_file.__name__, pfile))
        assert isinstance(basename, str)
        nbasename = basename.replace('0.0', '0p0')
        npfile = FileFolderTools.full_path(dirname + '/' + nbasename)

        print('[%s] new filename %s' % (self.process_file.__name__, npfile))
        os.rename(pfile, npfile)

    def process_bdir(self):
        # print('[process_bdir] processing %s' % self.bdir)
        print('[%s] processing %s' % (self.process_bdir.__name__, self.bdir))

        # foldername, mextension, method
        FileFolderTools.run_method_on_folder(self.bdir, '',
                                             ulgprocess.process_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do stuff')
    parser.add_argument('--ubdir', action='store', required=True,
                        help='Base directory')
    # parser.add_argument('--plot', action='store_true', required=False,
    #                     help='plot results')
    # parser.add_argument('--loc', action='store_true', help='location')
    args = parser.parse_args()

    ubdir = FileFolderTools.full_path(args.bdir)
    # uplot = args.plot

    foldername_arr = FileFolderTools.get_folder_arr(ubdir, 'Angle')
    print(foldername_arr)
    for folder in foldername_arr:
        ulgprocess = Process(ubdir)
        ulgprocess.process_bdir()
        assert isinstance(ubdir, str)
        newname = ubdir.replace('0.0', '0p0')
        os.rename(ubdir, newname)
