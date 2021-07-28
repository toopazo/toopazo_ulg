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
# import matplotlib.pyplot as plt
# import numpy as np
# from numba.cuda import stream
# import pandas as pd
import ctypes
import subprocess

# import threading
import multiprocessing
import queue
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
            _ = i
            # pname = 'multiprocessing_worker_%s' % ui
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
            _ = proc
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


if __name__ == '__main__':
    mtimeout = 1
    for ui in range(0, 5):
        marg = (ui,)
        ExternalProcess.exec_with_multiprocessing(
            ExternalProcess.multiprocessing_worker, marg, mtimeout)
