#!/usr/bin/env python

# import argparse
# import subprocess
from toopazo_utilities import FileFolderUtils
# import csv
from datetime import datetime, date, time
# import math
import matplotlib.pyplot as plt
import numpy as np
from toopazo_statistics import TimeseriesStats
from toopazo_ulgParser import UlgParser


class UlgPlotter:
    def __init__(self, plotdir):
        self.plotdir = plotdir

    @staticmethod
    def timestamp_to_datetime(x):
        xdt = []
        for tstamp in x:
            xdt.append(datetime.fromtimestamp(tstamp))
        x = xdt
        return x

    @staticmethod
    def nwindow_fcost(y):
        y = np.abs(y)
        v = np.mean(y)
        return v

    @staticmethod
    def get_jpgfile(fpath, ulgfile, csvname):
        ulgfile = FileFolderUtils.get_basename(ulgfile)
        filename = ulgfile.replace('.ulg', '_') + csvname + '.jpg'
        filename = fpath + '/' + filename
        return filename

    def plot_vehicle_attitude_0_deg(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax_arr = [ax0, ax1, ax2]
        fig.suptitle('Timeseries: vehicle_attitude_0_deg')

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2]
        ylabel_arr = ['Roll deg', 'Pitch deg', 'Yaw deg']
        PlotToFig.ax3_x1_y3(ax_arr, x, xlabel, y_arr, ylabel_arr)
        ax0.set_ylim([-30, 30])
        ax1.set_ylim([-30, 30])
        ax2.set_ylim([-30, 30])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_vehicle_rates_setpoint_0(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_vehicle_rates_setpoint_0(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        ax_arr = [ax0, ax1, ax2, ax3]
        fig = plt.figure()
        fig.suptitle('Timeseries: vehicle_rates_setpoint_0')

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['roll', 'pitch', 'yaw', 'throttle']
        PlotToFig.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        ax0.set_ylim([-90, 90])
        ax1.set_ylim([-90, 90])
        ax2.set_ylim([-90, 90])
        ax3.set_ylim([0, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_manual_control_setpoint_0(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_manual_control_setpoint_0(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        # fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        # ax_arr = [ax0, ax1, ax2]
        fig = plt.figure()
        fig.suptitle('Timeseries: manual_control_setpoint_0')
        ax = plt.gca()

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2, y3]
        ylabel = 'RC inputs'
        PlotToFig.ax1_x1_y4(ax, x, xlabel, y_arr, ylabel)
        ax.set_ylim([-1, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_vehicle_local_position_0(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3, y4, y5] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax_arr = [ax0, ax1, ax2]
        fig.suptitle('Timeseries: vehicle_local_position_0')

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2]
        ylabel_arr = ['x', 'y', 'z']
        PlotToFig.ax3_x1_y3(ax_arr, x, xlabel, y_arr, ylabel_arr)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_actuator_controls_0_0(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        ax_arr = [ax0, ax1, ax2, ax3]
        fig.suptitle('Timeseries: actuator_controls_0_0')

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['control[0]', 'control[1]', 'control[2]', 'control[3]']
        PlotToFig.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        ax0.set_ylim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax3.set_ylim([0, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_actuator_outputs_0(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, tmpdir)
        x = UlgPlotter.timestamp_to_datetime(x)

        fig = plt.figure()
        fig.suptitle('Timeseries: actuator_outputs_0')
        ax = plt.gca()

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2, y3, y4, y5, y6, y7]
        ylabel = 'actuator_outputs_0'
        PlotToFig.ax1_x1_y8(ax, x, xlabel, y_arr, ylabel)
        ax.set_ylim([700, 2200])

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname+"_a")
        PlotToFig.savecf(jpgfile, False)

        # Next figure

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        fig.suptitle('Timeseries: actuator_outputs_0')
        ax_arr = [ax0, ax1, ax2, ax3]
        xlabel = 'timestamp'

        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['m1', 'm2', 'm3', 'm4']
        PlotToFig.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        ax0.set_ylim([700, 2200])
        ax1.set_ylim([700, 2200])
        ax2.set_ylim([700, 2200])
        ax3.set_ylim([700, 2200])

        y_arr = [y5, y4, y7, y6]
        ylabel_arr = ['m1, m6', 'm2, m5', 'm3, m8', 'm4, m7']
        PlotToFig.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname+"_b")
        PlotToFig.savecf(jpgfile, False)

    def plot_nwindow_hover_pos(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3, y4, y5] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, tmpdir)

        # 1 sec = 10**6 microsec
        # Actual SPS in log files is approx 10
        window = 10*3
        lenx = len(x)
        if window > lenx:
            window = lenx
            print('[plot_nwindow_hover_pos] window %s < len(x) %s ' %
                  (window, lenx))

        nmax = len(y0) - 1
        ilast = nmax - window + 1
        x_window = x[0:ilast+1]
        y0_window = TimeseriesStats.apply_to_window(y0, np.std, window)
        y1_window = TimeseriesStats.apply_to_window(y1, np.std, window)
        y2_window = TimeseriesStats.apply_to_window(y2, np.std, window)
        y3_window = np.add(y0_window, y1_window, y2_window)

        argmin_y3_window = int(np.argmin(y3_window))
        min_y3_window = y3_window[argmin_y3_window]
        min_x = x[argmin_y3_window]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        arg = 'Timeseries: window = %s, min(std) = %s, time[min(std)] = %s' % \
              (window, round(min_y3_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel = 'timestamp'
        y_arr = [y0, y1, y2]
        ylabel = 'x, y, z'
        PlotToFig.ax1_x1_y3(ax0, x, xlabel, y_arr, ylabel)

        xlabel = 'timestamp'
        y_arr = [y0_window, y1_window, y2_window, y3_window]
        ylabel = 'std window'
        PlotToFig.ax1_x1_y4(ax1, x_window, xlabel, y_arr, ylabel)

        i0 = argmin_y3_window
        il = argmin_y3_window + window

        xlabel = 'timestamp'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il]]
        ylabel = 'x, y, z'
        PlotToFig.ax1_x1_y3(ax2, x[i0:il], xlabel, y_arr, ylabel)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, tmpdir)
        xlabel = 'timestamp'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel = 'actuator_outputs_0'
        PlotToFig.ax1_x1_y8(ax3, x[i0:il], xlabel, y_arr, ylabel)

        csvname = 'hover_nwindow_pos'
        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

    def plot_nwindow_hover_vel(self, ulgfile, tmpdir):
        [csvname, x, y0, y1, y2, y3, y4, y5] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, tmpdir)

        # 1 sec = 10**6 microsec
        # Actual SPS in log files is approx 10
        wind    ow = 10*2
        lenx = len(x)
        if window > lenx:
            window = lenx
            print('[plot_nwindow_hover_vel] window %s < len(x) %s ' %
                  (window, lenx))
            raise RuntimeError

        nmax = len(x) - 1
        ilast = nmax - window + 1
        x_window = x[0:ilast+1]
        fcost = UlgPlotter.nwindow_fcost
        y3_window = TimeseriesStats.apply_to_window(y3, fcost, window)
        y4_window = TimeseriesStats.apply_to_window(y4, fcost, window)
        y5_window = TimeseriesStats.apply_to_window(y5, fcost, window)
        y6_window = np.add(y3_window, y4_window, y5_window)

        argmin_y6_window = int(np.argmin(y6_window))
        min_y6_window = y6_window[argmin_y6_window]
        min_x = x[argmin_y6_window]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        arg = 'Timeseries: window = %s, min(fcost) = %s, time[min(fcost)] = %s'\
              % (window, round(min_y6_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel = 'timestamp'
        y_arr = [y3, y4, y5]
        ylabel = 'vx, vy, vz'
        PlotToFig.ax1_x1_y3(ax0, x, xlabel, y_arr, ylabel)

        xlabel = 'timestamp'
        y_arr = [y3_window, y4_window, y5_window, y6_window]
        ylabel = 'fcost'
        PlotToFig.ax1_x1_y4(ax1, x_window, xlabel, y_arr, ylabel)

        i0 = argmin_y6_window
        il = argmin_y6_window + window

        xlabel = 'timestamp'
        y_arr = [y3[i0:il], y4[i0:il], y5[i0:il]]
        ylabel = 'vx, vy, vz'
        PlotToFig.ax1_x1_y3(ax2, x[i0:il], xlabel, y_arr, ylabel)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, tmpdir)

        xlabel = 'timestamp'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel = 'actuator variables'
        PlotToFig.ax1_x1_y8(ax3, x[i0:il], xlabel, y_arr, ylabel)
        txt = 'std green %s' % round(float(np.std(y0[i0:il])), 2)
        ax3.annotate(txt, xy=(0.05, 0.8), xycoords='axes fraction')

        csvname = 'hover_nwindow_vel'
        jpgfile = self.get_jpgfile(self.plotdir, ulgfile, csvname)
        PlotToFig.savecf(jpgfile, False)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()


class PlotToFig:
    def __init__(self):
        pass

    @staticmethod
    def ax1_x1_y8(ax, x, xlabel, y_arr, ylabel):
        # ax.hold(True)
        ax.grid(True)
        ax.plot(x, y_arr[0], color='red')
        ax.plot(x, y_arr[1], color='green')
        ax.plot(x, y_arr[2], color='blue')
        ax.plot(x, y_arr[3], color='black')
        ax.plot(x, y_arr[4], color='green', linestyle='dashed')
        ax.plot(x, y_arr[5], color='red', linestyle='dashed')
        ax.plot(x, y_arr[6], color='black', linestyle='dashed')
        ax.plot(x, y_arr[7], color='blue', linestyle='dashed')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        # ax.ticklabel_format(useOffset=False)
        #   I was getting the error
        #   "This method only works with the ScalarFormatter.")
        #   AttributeError: This method only works with the ScalarFormatter.
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=4)
        # plt.xticks(rotation=45)
        # ax.set_xticklabels(getLabels(s, t), rotation=20)

    @staticmethod
    def ax1_x1_y4(ax, x, xlabel, y_arr, ylabel):
        # ax.hold(True)
        ax.grid(True)
        ax.plot(x, y_arr[0], color='red')
        ax.plot(x, y_arr[1], color='green')
        ax.plot(x, y_arr[2], color='blue')
        ax.plot(x, y_arr[3], color='black')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.ticklabel_format(useOffset=False)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y3(ax, x, xlabel, y_arr, ylabel):
        # ax.hold(True)
        ax.grid(True)
        ax.plot(x, y_arr[0], color='red')
        ax.plot(x, y_arr[1], color='green')
        ax.plot(x, y_arr[2], color='blue')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.ticklabel_format(useOffset=False)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=4)

    @staticmethod
    def ax3_x1_y3(ax_arr, x, xlabel, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y_arr[0])
        ax_arr[0].set(xlabel=xlabel, ylabel=ylabel_arr[0])
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y_arr[1])
        ax_arr[1].set(xlabel=xlabel, ylabel=ylabel_arr[1])
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y_arr[2])
        ax_arr[2].set(xlabel=xlabel, ylabel=ylabel_arr[2])
        ax_arr[2].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y_arr[0])
        ax_arr[0].set(xlabel=xlabel, ylabel=ylabel_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y_arr[1])
        ax_arr[1].set(xlabel=xlabel, ylabel=ylabel_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y_arr[2])
        ax_arr[2].set(xlabel=xlabel, ylabel=ylabel_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

        # ax_arr[3].hold(True)
        ax_arr[3].grid(True)
        ax_arr[3].plot(x, y_arr[3])
        ax_arr[3].set(xlabel=xlabel, ylabel=ylabel_arr[3])
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)

    @staticmethod
    def savecf(filename, close):
        # plt.show()
        plt.draw()

        # # Make fig the current figure
        # assert isinstance(fig, plt.figure())
        # plt.figure(fig.number)

        # Save and close current figure
        print('[savecf] filename %s' %
              FileFolderUtils.get_basename(filename))
        plt.savefig(filename)

        if close:
            plt.clf()
            plt.close()


if __name__ == '__main__':
    pass
