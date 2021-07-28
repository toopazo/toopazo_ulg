#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os
import argparse
import time


__author__ = 'toopazo'


class FileFolderTools:
    def __init__(self):
        pass

    @staticmethod
    def chdir(fpath):
        if FileFolderTools.is_folder(fpath):
            os.chdir(ubdir)
        else:
            print('[chdir] Path %s is not a folder' % fpath)
            raise RuntimeError

    @staticmethod
    def get_cwd():
        return os.getcwd()

    @staticmethod
    def is_file(fpath):
        return os.path.isfile(fpath)

    @staticmethod
    def is_folder(fpath):
        return os.path.isdir(fpath)

    @staticmethod
    def full_path(fpath):
        fpath = os.path.normcase(fpath)
        fpath = os.path.normpath(fpath)
        fpath = os.path.realpath(fpath)
        return fpath

    @staticmethod
    def get_folder_arr(folderpath, pattern):
        # targetfolder = FileFolderTools.full_path(targetfolder)
        folder_arr = []
        for item in os.listdir(folderpath):
            if os.path.isdir(os.path.join(folderpath, item)):
                if pattern in item:
                    folder_arr.append(item)
        folder_arr.sort()

        return folder_arr

    @staticmethod
    def get_file_arr(fpath, extension):
        # fpath = FileFolderTools.full_path(fpath)
        file_arr = []
        for item in os.listdir(fpath):
            if os.path.isfile(os.path.join(fpath, item)):
                # item, file_extension = os.path.splitext(item)
                # if file_extension == mextension:
                #     res_arr.append(item)
                if extension in item:
                    # item = FileFolderTools.full_path(item)
                    item = fpath + '/' + item
                    # print('[get_file_arr] item %s' % item)
                    file_arr.append(item)
        file_arr.sort()

        # nfiles = len(res_arr)
        # print("[get_file_arr] %s \"%s\" files were found at %s"
        #       % (nfiles, mextension, fpath))

        return file_arr

    @staticmethod
    def get_file_split(fpath):
        #  os.path.split(path)
        #
        # Split the pathname path into a pair, (head, tail) where tail is the
        # last pathname component and head is everything leading up to that.
        # The tail part will never contain a slash; if path ends in a slash,
        # tail will be empty. If there is no slash in path, head will be empty.
        # If path is empty, both head and tail are empty. Trailing slashes are
        # stripped from head unless it is the root (one or more slashes only).
        # In all cases, join(head, tail) returns a path to the same location as
        # path (but the strings may differ). Also see the functions dirname()
        # and basename().
        if FileFolderTools.is_file(fpath):
            (head, tail) = os.path.split(fpath)
            return [head, tail]
        else:
            arg = '[get_file_split] Path %s is not a file' % fpath
            print(arg)
            return arg  # raise RuntimeError(arg)

    @staticmethod
    def get_file_basename(fpath):
        if FileFolderTools.is_file(fpath):
            # Return the base name of pathname fpath. This is the second
            # element of the pair returned by passing fpath to the function
            # split().
            return os.path.basename(fpath)
        else:
            arg = '[get_file_basename] Path %s is not a file' % fpath
            print(arg)
            return arg  # raise RuntimeError(arg)

    @staticmethod
    def run_method_on_folder(fpath, extension, method):
        fpath = FileFolderTools.full_path(fpath)

        if not FileFolderTools.is_folder(fpath):
            print("[run_method_on_folder] Path %s is not a folder" % fpath)
            print("**********************************************************")
            print("Done")
            return

        print('\n')
        print("[run_method_on_folder] Running %s on folder %s "
              % (method.__name__, fpath))
        print("**********************************************************")

        # get ".mextension" files and apply "method" over every file
        file_arr = FileFolderTools.get_file_arr(
            fpath=fpath, extension=extension)

        # print('[run_method_on_folder] %s' % res_arr)

        for filename in file_arr:
            # filepath = FileFolderTools.full_path(fpath + '/' + filename)
            # directory, filename = os.path.split(filepath)
            # print("[run_method_on_folder] filename %s .." % filename)
            filepath = filename
            print("[run_method_on_folder] File basename %s "
                  % FileFolderTools.get_file_basename(filepath))
            method(filepath)
            # print("Next file ")

            print("**********************************************************")

        print("Done")

    @staticmethod
    def delete_file(fpath):
        if not FileFolderTools.is_file(fpath):
            arg = '[delete_file] Path %s is not a file' % fpath
            print(arg)
            return arg  # raise RuntimeError(arg)
        os.remove(fpath)

    @staticmethod
    def clear_folders(fpath):
        # fpath = 'logs'
        fpath = FileFolderTools.full_path(fpath)

        file_arr = os.listdir(fpath)
        for filename in file_arr:
            filepath = fpath + '/' + filename
            filepath = FileFolderTools.full_path(filepath)
            print('[clear_folders] removing %s' % filepath)
            os.remove(filepath)

        # fpath = 'events'
        # fpath = FileFolderTools.full_path(fpath)
        #
        # res_arr = os.listdir(fpath)
        # for filename in res_arr:
        #     mfilepath = fpath + '/' + filename
        #     mfilepath = FileFolderTools.full_path(mfilepath)
        #     print('[clear_folders] removing %s' % mfilepath)
        #     os.remove(mfilepath)

    @staticmethod
    def get_file_info(fpath):
        if not FileFolderTools.is_file(fpath):
            arg = '[get_file_info] Path %s is not a file' % fpath
            print(arg)
            return arg  # raise RuntimeError(arg)

        [head, tail] = FileFolderTools.get_file_split(fpath)
        print('[get_file_info] head %s , tail %s' % (head, tail))

        last_access = os.path.getatime(fpath)
        last_modified = os.path.getmtime(fpath)
        size_bytes = os.path.getsize(fpath)

        print('[get_file_info] last_access %s' % last_access)
        print('[get_file_info] last_modified %s' % last_modified)
        print('[get_file_info] size_bytes %s' % size_bytes)

    @staticmethod
    def rename_file_in_folder(oldstr, newstr, folder, extension):
        print('[rename_file_in_folder] Running on folder %s' % folder)
        file_arr = FileFolderTools.get_file_arr(folder, extension)
        for oldfpath in file_arr:
            # print('[rename_file_in_folder] oldfpath %s' % oldfpath)
            [head, tail] = FileFolderTools.get_file_split(oldfpath)
            if oldstr in tail:
                newtail = tail.replace(oldstr, newstr)
                newfpath = head + "/" + newtail
                os.rename(oldfpath, newfpath)


if __name__ == '__main__':
    # Run this part with
    # python -m toopazo_tools.FileFolderTools

    parser = argparse.ArgumentParser(
        description='Parse, process and plot .ulg files')
    parser.add_argument('--test', action='store_true', required=False,
                        help='Run unit test on test/ folder')
    parser.add_argument('--run', action='store', required=False,
                        help='Run selected method')
    args = parser.parse_args()

    testflag = args.test
    runmethod = args.run

    if testflag:
        ubdir = '/home/tzo4/Dropbox/tomas/pennState/avia/software/toopazo_tools'

        FileFolderTools.chdir(ubdir)

        targetfolder = FileFolderTools.get_cwd()
        print('FileFolderTools.get_cwd() => %s' % targetfolder)

        res = FileFolderTools.is_file(targetfolder)
        print('FileFolderTools.is_file() => %s' % res)

        res = FileFolderTools.is_folder(targetfolder)
        print('FileFolderTools.is_folder() => %s' % res)

        targetfolder = FileFolderTools.full_path(targetfolder)
        print('FileFolderTools.full_path() => %s' % targetfolder)

        mpattern = ''
        res_arr = FileFolderTools.get_folder_arr(targetfolder, mpattern)
        print('FileFolderTools.get_folder_arr() => %s' % res_arr)

        mextension = ''
        res_arr = FileFolderTools.get_file_arr(targetfolder, mextension)
        print('FileFolderTools.get_file_arr() => %s' % res_arr)

        # Select test folder
        testfolder = FileFolderTools.full_path(targetfolder + '/tests')
        print('testfolder %s' % testfolder)

        # Delete test files
        mextension = '.txt'
        FileFolderTools.run_method_on_folder(
            testfolder, mextension, FileFolderTools.delete_file)

        time.sleep(3)

        # Write file to
        for i in range(1, 10):
            mfilepath = targetfolder + '/tests/test_' + str(i) + '.txt'
            mfilepath = FileFolderTools.full_path(mfilepath)
            fd = open(mfilepath, 'w')
            fd.write('blah blah ' + str(i))
            fd.close()

        mextension = ''
        FileFolderTools.run_method_on_folder(
            testfolder, mextension, FileFolderTools.get_file_info)

        time.sleep(3)

        FileFolderTools.rename_file_in_folder(
            oldstr='test_', newstr='testfile_',
            folder=testfolder, extension='.txt')

    if runmethod is not None:
        umethod = getattr(FileFolderTools, runmethod)
        # tf = '/home/tzo4/Dropbox/tomas/pennState/avia/firefly_logBook/' \
        #      '2020-12-22_firefly_mixer'
        # umethod(oldstr='.txt', newstr='.ars', folder=tf, extension='.txt')
