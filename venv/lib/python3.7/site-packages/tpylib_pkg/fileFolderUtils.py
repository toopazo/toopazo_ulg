#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os


__author__ = 'toopazo'


class FileFolderUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_currfolder():
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
        # targetfolder = FileFolderUtils.full_path(targetfolder)
        folder_arr = []
        for item in os.listdir(folderpath):
            if os.path.isdir(os.path.join(folderpath, item)):
                if pattern in item:
                    folder_arr.append(item)
        folder_arr.sort()

        return folder_arr

    @staticmethod
    def get_file_arr(fpath, extension):
        # fpath = FileFolderUtils.full_path(fpath)
        file_arr = []
        for item in os.listdir(fpath):
            if os.path.isfile(os.path.join(fpath, item)):
                # item, file_extension = os.path.splitext(item)
                # if file_extension == mextension:
                #     res_arr.append(item)
                if extension in item:
                    file_arr.append(item)
        file_arr.sort()

        # nfiles = len(res_arr)
        # print("[get_file_arr] %s \"%s\" files were found at %s"
        #       % (nfiles, mextension, fpath))

        return file_arr

    @staticmethod
    def get_basename(fpath):
        # Return the base name of pathname fpath. This is the second element of
        # the pair returned by passing fpath to the function split().
        return os.path.basename(fpath)

    @staticmethod
    def run_method_on_folder(fpath, extension, method):
        fpath = FileFolderUtils.full_path(fpath)

        if not FileFolderUtils.is_folder(fpath):
            print("[run_method_on_folder] fpath %s is not a folder" % fpath)
            print("**********************************************************")
            print("Done")
            return

        print("[run_method_on_folder] fpath %s " % fpath)
        print("**********************************************************")

        # get ".mextension" files and apply "method" over every file
        file_arr = FileFolderUtils.get_file_arr(
            fpath=fpath, extension=extension)

        # print('[run_method_on_folder] %s' % res_arr)

        for filename in file_arr:
            filepath = FileFolderUtils.full_path(fpath + '/' + filename)
            # directory, filename = os.path.split(filepath)
            # print("[run_method_on_folder] filename %s .." % filename)
            print("[run_method_on_folder] filepath %s " % filepath)
            method(filepath)
            # print("Next file ")

            print("**********************************************************")

        print("Done")

    @staticmethod
    def clear_folders(fpath):
        # fpath = 'logs'
        fpath = FileFolderUtils.full_path(fpath)

        file_arr = os.listdir(fpath)
        for filename in file_arr:
            filepath = fpath + '/' + filename
            filepath = FileFolderUtils.full_path(filepath)
            print('[clear_folders] removing %s' % filepath)
            os.remove(filepath)

        # fpath = 'events'
        # fpath = FileFolderUtils.full_path(fpath)
        #
        # res_arr = os.listdir(fpath)
        # for filename in res_arr:
        #     mfilepath = fpath + '/' + filename
        #     mfilepath = FileFolderUtils.full_path(mfilepath)
        #     print('[clear_folders] removing %s' % mfilepath)
        #     os.remove(mfilepath)

    @staticmethod
    def get_file_info(fpath):
        last_access = os.path.getatime(fpath)
        last_modified = os.path.getmtime(fpath)
        size_bytes = os.path.getsize(fpath)

        print('[get_file_info] last_access %s' % last_access)
        print('[get_file_info] last_modified %s' % last_modified)
        print('[get_file_info] size_bytes %s' % size_bytes)


if __name__ == '__main__':
    # print('Given a path')

    targetfolder = FileFolderUtils.get_currfolder()
    print('FileFolderUtils.get_currfolder() => %s' % targetfolder)

    targetfolder = os.path.normpath(targetfolder + '/..')

    res = FileFolderUtils.is_file(targetfolder)
    print('FileFolderUtils.is_file() => %s' % res)

    res = FileFolderUtils.is_folder(targetfolder)
    print('FileFolderUtils.is_folder() => %s' % res)

    targetfolder = FileFolderUtils.full_path(targetfolder)
    print('FileFolderUtils.full_path() => %s' % targetfolder)

    mpattern = ''
    res_arr = FileFolderUtils.get_folder_arr(targetfolder, mpattern)
    print('FileFolderUtils.get_folder_arr() => %s' % res_arr)

    mextension = ''
    res_arr = FileFolderUtils.get_file_arr(targetfolder, mextension)
    print('FileFolderUtils.get_file_arr() => %s' % res_arr)

    bname = FileFolderUtils.get_basename(targetfolder)
    print('FileFolderUtils.get_basename() => %s' % bname)

    # Write file to
    testfolder = FileFolderUtils.full_path(targetfolder + '/tests')
    for i in range(0, 3):
        mfilepath = targetfolder + '/tests/test_' + str(i) + '.txt'
        mfilepath = FileFolderUtils.full_path(mfilepath)
        fd = open(mfilepath, 'w')
        fd.write(str(i))
        fd.close()

    mextension = ''
    FileFolderUtils.run_method_on_folder(
        testfolder, mextension, FileFolderUtils.get_file_info)

    mextension = ''
    FileFolderUtils.run_method_on_folder(
        'asdasd', mextension, FileFolderUtils.get_file_info)

    # FileFolderUtils.clear_folders(testfolder)

