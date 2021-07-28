#!/usr/bin/env python

import numpy as np
from scipy import signal


class TimeseriesTools:

    @staticmethod
    def test_window(arr, n):
        return np.std(TimeseriesTools.rolling_window(arr, n), 1)

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
            sarr = TimeseriesTools.get_window(arr, i, nwindow)
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
        TimeseriesTools.apply_to_window(marr, TimeseriesTools.add_sarr, mwindow)

    @staticmethod
    def common_statistics(x_arr, verbose):
        x_arr = np.array(x_arr)
        x_nsamples = len(x_arr)

        # Mean
        x_mean = np.mean(x_arr)
        # Standard deviation
        x_std = np.std(x_arr)
        # Signed deviation
        x_sgndev = x_arr - x_mean
        # Unsigned deviation
        x_usgndev = np.abs(x_sgndev)
        # Mean signed deviation
        x_meansgndev = np.mean(x_sgndev)
        # Max unsigned deviation
        x_maxusgndev = np.max(x_usgndev)
        # Min unsigned deviation
        x_minusgndev = np.min(x_usgndev)

        rdict = {'x_nsamples': x_nsamples,
                 'x_mean': x_mean,
                 'x_std': x_std,
                 'x_meansgndev': x_meansgndev,
                 'x_minusgndev': x_minusgndev,
                 'x_maxusgndev': x_maxusgndev,
                 }

        if verbose:
            for key, value in rdict.items():
                print('[common_statistics] rdict[%s] = %s' % (key, value))

        return rdict

    @staticmethod
    def time_statistics(t_arr, verbose):
        t_arr = np.array(t_arr)
        t_nsamples = len(t_arr)

        # dt = delta times between samples
        t_diff = np.diff(t_arr)
        # Mean
        t_dt_mean = np.mean(t_diff)
        # Standard deviation
        t_dt_std = np.std(t_diff)
        # Signed deviation
        t_dt_sgndev = t_diff - t_dt_mean
        # Unsigned deviation
        t_dt_usgndev = np.abs(t_diff - t_dt_mean)
        # Mean signed deviation
        t_dt_meansgndev = np.mean(t_dt_sgndev)
        # Max unsigned deviation
        t_dt_maxusgndev = np.max(t_dt_usgndev)
        # Min unsigned deviation
        t_dt_minusgndev = np.min(t_dt_usgndev)

        t_window = t_arr[-1] - t_arr[0]
        t_tferror = (t_arr[0] + t_dt_mean*(t_nsamples-1)) - t_arr[-1]

        rdict = {'t_nsamples': t_nsamples,
                 't_dt_mean': t_dt_mean,
                 't_dt_std': t_dt_std,
                 't_dt_meansgndev': t_dt_meansgndev,
                 't_dt_minusgndev': t_dt_minusgndev,
                 't_dt_maxusgndev': t_dt_maxusgndev,
                 't_window': t_window,
                 't_tferror': t_tferror
                 }

        if verbose:
            for key, value in rdict.items():
                print('[time_statistics] rdict[%s] = %s' % (key, value))

        return rdict

    @staticmethod
    def resample(t1_arr, x1_arr, t2_arr, tolkey, tolval, verbose):
        t1_arr = np.array(t1_arr)
        t2_arr = np.array(t2_arr)
        x1_arr = np.array(x1_arr)

        t1_rdict = TimeseriesTools.time_statistics(t1_arr, False)
        t1_nsamples = t1_rdict['t_nsamples']
        t1_dt_mean = t1_rdict['t_dt_mean']

        t2_rdict = TimeseriesTools.time_statistics(t2_arr, False)
        # t2_nsamples = t2_rdict['t_nsamples']
        t2_dt_mean = t2_rdict['t_dt_mean']

        x1_nsamples = len(x1_arr)

        if x1_nsamples != t1_nsamples:
            print('[resample] t1_arr: x1_nsamples = %s != t1_nsamples = %s'
                  % (x1_nsamples, t1_nsamples))
            return x1_arr

        if t1_rdict[tolkey] > tolval or t2_rdict[tolkey] > tolval:
            print('[resample] Tolerance error in t1_arr or t2_arr')

            print('[resample] t1_arr:')
            TimeseriesTools.time_statistics(t1_arr, True)

            print('[resample] t2_arr:')
            TimeseriesTools.time_statistics(t2_arr, True)
            return x1_arr

        # for a typical .ulg file scale_factor = 245.687605706
        scale_factor = t1_dt_mean / t2_dt_mean

        # scale_factor = int(round(scale_factor))
        # Number of samples that t1_arr should have if it were to be
        # sampled at the frequency of t2_arr
        t1_nsamples_expected = scale_factor * t1_nsamples
        t1_nsamples_expected = int(round(t1_nsamples_expected))

        # x2_arr = signal.resample(x1_arr, t1_nsamples_expected, t=None)
        (rx1_arr, rt1_arr) = \
            signal.resample(x1_arr, t1_nsamples_expected, t=t1_arr)

        if verbose:
            print('[resample] t1_arr:')
            TimeseriesTools.time_statistics(t1_arr, True)

            print('[resample] x1_arr:')
            print('[resample] x1_nsamples %s' % len(x1_arr))

            print('[resample] t2_arr:')
            TimeseriesTools.time_statistics(t2_arr, True)

            print('[resample] scale_factor %s' % scale_factor)

            print('[resample] rt1_arr:')
            TimeseriesTools.time_statistics(rt1_arr, True)

            print('[resample] rx1_arr:')
            print('[resample] rx1_nsamples %s' % len(rx1_arr))

        return [rt1_arr, rx1_arr]


if __name__ == '__main__':
    TimeseriesTools.test_iterate_window()
