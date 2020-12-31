#!/usr/bin/env python

import numpy as np


class TimeseriesStats:

    @staticmethod
    def test_window(arr, n):
        return np.std(TimeseriesStats.rolling_window(arr, n), 1)

    @staticmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def get_window(arr, i, nwindow):
        # size = np.shape(arr)
        # nmax = len(arr) - 1
        ifirst = i
        ilast = i + nwindow
        # if ilast > nmax:
        #     ilast = nmax
        #     sarr = arr[ifirst:ilast]
        # else:
        sarr = arr[ifirst:ilast]
        return sarr

    @staticmethod
    def apply_to_window(arr, fun, nwindow):

        nmax = len(arr) - 1

        # arg = '[apply_to_window] %s' % arr
        # print(arg)
        arg = '[apply_to_window] nwindow %s' % nwindow
        print(arg)
        arg = '[apply_to_window] nmax %s' % nmax
        print(arg)

        ilast = nmax - nwindow + 1
        res_arr = []
        for i in range(0, ilast + 1):
            sarr = TimeseriesStats.get_window(arr, i, nwindow)
            res = fun(sarr)
            res_arr.append(res)
            # arg = '[apply_to_window] ui %s, sarr %s, res %s' % (ui, sarr, res)
            # print(arg)
        return np.array(res_arr)

    @staticmethod
    def add_sarr(sarr):
        return np.sum(sarr)

    @staticmethod
    def add_abs_sarr(sarr):
        return np.std(sarr)

    @staticmethod
    def test_iterate_window():
        marr = np.arange(0, 15)
        mwindow = 5
        TimeseriesStats.apply_to_window(marr, TimeseriesStats.add_sarr, mwindow)


if __name__ == '__main__':
    TimeseriesStats.test_iterate_window()
