#!/usr/bin/Python
# -*- coding: utf-8 -*-

# from dbus import exceptions
# from datetime import datetime
# import dateutil.parser
# import datetime
# import os
# import sys
# from  pprint import pprint

# import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
# from numba.cuda import stream
import pandas as pd
import ctypes
import subprocess

# import threading
import multiprocessing
# import queue
import time
import os


__author__ = 'toopazo'


class ExternalProcess:
    def __init__(self):
        pass

    # Define a function for the thread
    @staticmethod
    def multiprocessing_worker(num):
        cpname = multiprocessing.current_process().name
        arg = "[%s] Starting " % cpname
        print(arg)

        print('[%s] given argument is %s' % (cpname, num))
        print('[%s] PID %s' % (cpname, os.getpid()))

        if num == 1:
            ExternalProcess.do_segfault()

        time.sleep(3)

        arg = "[%s] Ending " % cpname
        print(arg)
        return

    @staticmethod
    def exec_with_multiprocessing(pfnct, pargs, timeout):
        p_arr = []
        pqueue = multiprocessing.Queue()
        pnum = 1
        for i in range(pnum):
            # pname = 'multiprocessing_worker_%s' % i
            pname = pfnct.__name__
            proc = multiprocessing.Process(name=pname,
                                           target=pfnct,
                                           args=(pqueue,) + pargs)
            # To wait until a daemon thread finishes use the join() method
            # proc.daemon = True     # setDaemon(True)
            p_arr.append(proc)

            # arg = "[exec_with_multiprocessing] About to launch %s" % proc.name
            # print(arg)

            proc.start()

        # eventtime.sleep(1)

        reply = None
        for proc in p_arr:
            # arg = "[exec_with_multiprocessing] %s isAlive() %s" % \
            #       (proc.name, proc.is_alive())
            # print(arg)

            try:
                reply = pqueue.get(timeout=timeout)
            except queue.Empty:
                # reply = 'No reply from %s' % pname
                reply = None

            # arg = "[exec_with_multiprocessing] reply %s" % reply
            # print(arg)

        # eventtime.sleep(5)
        for proc in p_arr:
            proc.join()
        # print('[exec_with_multiprocessing] All process_file joined')

        return reply

    @staticmethod
    def exec_with_subprocess(cmd_seq):
        try:
            arg = "[exec_with_subprocess] cmd_seq = %s" % cmd_seq
            print(arg)

            byte_string = subprocess.check_output(
                cmd_seq,
                stderr=subprocess.STDOUT,
                # stdin=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                shell=True)        # stderr=subprocess.STDOUT,

            arg = "[exec_with_subprocess] len(byte_string)  %s" % \
                  len(byte_string)
            print(arg)
        except subprocess.CalledProcessError:
            byte_string = 'Exception subprocess.CalledProcessError'

        return byte_string

    @staticmethod
    def do_segfault():
        # https://codegolf.stackexchange.com/questions/4399
        # /shortest-code-that-raises-a-sigsegv
        print('[do_segfault] bye bye !!')
        ctypes.string_at(0)  # segmentation fault


class FileFolderUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_cfolder():
        return os.getcwd()

    @staticmethod
    def is_file(infile):
        return os.path.isfile(infile)

    @staticmethod
    def is_folder(infile):
        return os.path.isdir(infile)

    @staticmethod
    def full_path(infile):
        infile = os.path.normcase(infile)
        infile = os.path.normpath(infile)
        infile = os.path.realpath(infile)
        return infile

    @staticmethod
    def get_foldername_arr(folderpath, pattern):
        # folderpath = FileFolderUtils.full_path(folderpath)
        foldername_arr = []
        for item in os.listdir(folderpath):
            if os.path.isdir(os.path.join(folderpath, item)):
                if pattern in item:
                    foldername_arr.append(item)
        foldername_arr.sort()

        return foldername_arr

    @staticmethod
    def get_filename_arr(folderpath, extension):
        # folderpath = FileFolderUtils.full_path(folderpath)
        filename_arr = []
        for item in os.listdir(folderpath):
            if os.path.isfile(os.path.join(folderpath, item)):
                # item, file_extension = os.path.splitext(item)
                # if file_extension == extension:
                #     filename_arr.append(item)
                if extension in item:
                    filename_arr.append(item)
        filename_arr.sort()

        # nfiles = len(filename_arr)
        # print("[get_filename_arr] %s \"%s\" files were found at %s"
        #       % (nfiles, extension, folderpath))

        return filename_arr

    @staticmethod
    def get_basename(path):
        # Return the base name of pathname path. This is the second element of
        # the pair returned by passing path to the function split().
        return os.path.basename(path)

    @staticmethod
    def run_method_on_folder(foldername, extension, method):
        folderpath = FileFolderUtils.full_path(foldername)

        print("[run_method_on_folder] folderpath %s " % folderpath)
        print("**********************************************************")

        # get ".extension" files and apply "method" over every file
        filename_arr = FileFolderUtils.get_filename_arr(
            folderpath=folderpath, extension=extension)

        # print('[run_method_on_folder] %s' % filename_arr)

        for filename in filename_arr:
            # filepath = os.path.abspath(filename)
            filepath = FileFolderUtils.full_path(folderpath + '/' + filename)
            # directory, filename = os.path.split(filepath)
            print("[run_method_on_folder] Processing filename %s .." % filename)
            print("[run_method_on_folder] Processing filepath %s .." % filepath)
            method(filepath)
            # print("Next file ")

            print("**********************************************************")

        print("Done")

    @staticmethod
    def clear_folders(foldername):
        # foldername = 'logs'
        folderpath = FileFolderUtils.full_path(foldername)

        filename_arr = os.listdir(folderpath)
        for filename in filename_arr:
            filepath = folderpath + '/' + filename
            filepath = FileFolderUtils.full_path(filepath)
            print('[clear_folders] removing %s' % filepath)
            os.remove(filepath)

        # folderpath = 'events'
        # folderpath = FileFolderUtils.full_path(folderpath)
        #
        # filename_arr = os.listdir(folderpath)
        # for filename in filename_arr:
        #     filepath = folderpath + '/' + filename
        #     filepath = FileFolderUtils.full_path(filepath)
        #     print('[clear_folders] removing %s' % filepath)
        #     os.remove(filepath)


if __name__ == '__main__':
    ExternalProcess.exec_with_multiprocessing(5, ExternalProcess.multiprocessing_worker)
