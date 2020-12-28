#!/usr/bin/env python

from datetime import datetime   # , date, time
import matplotlib.pyplot as plt
import numpy as np

# from tpylib_pkg.fileFolderUtils import FileFolderUtils
from tpylib_pkg.statistics import TimeseriesStats
from ulg_parser import UlgParser
from ulg_plot_figures import UlgPlotFigures


class UlgPlotBasics:
    def __init__(self, logdir, tmpdir, plotdir):
        self.logdir = logdir
        self.tmpdir = tmpdir
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

    def vehicle_attitude_0_deg(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(3, 1)
        fig.suptitle('Timeseries: vehicle_attitude_0_deg')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2]
        ylabel_arr = ['Roll deg', 'Pitch deg', 'Yaw deg']
        UlgPlotFigures.ax3_x1_y3(ax_arr, x, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-30, 30])
        # ax1.set_ylim([-30, 30])
        # ax2.set_ylim([-30, 30])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def vehicle_rates_setpoint_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5] = \
            UlgParser.get_vehicle_rates_setpoint_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: vehicle_rates_setpoint_0')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2, y5]
        ylabel_arr = ['roll', 'pitch', 'yaw', 'thrust_body']
        UlgPlotFigures.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-90, 90])
        # ax1.set_ylim([-90, 90])
        # ax2.set_ylim([-90, 90])
        # ax3.set_ylim([0, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def manual_control_setpoint_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_manual_control_setpoint_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        # fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        # ax_arr = [ax0, ax1, ax2]
        # fig = plt.figure()
        # ax = plt.gca()
        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(1, 1)
        fig.suptitle('Timeseries: manual_control_setpoint_0')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2, y3]
        ylabel = 'RC inputs'
        UlgPlotFigures.ax1_x1_y4(ax_arr, x, xlabel, y_arr, ylabel)
        # ax.set_ylim([-1, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def vehicle_local_position_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(3, 1)
        fig.suptitle('Timeseries: vehicle_local_position_0')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2, y3, y4, y5]
        ylabel_arr = ['x m, vx m/s', 'y m, vy m/s', 'z m, vz m/s']
        # UlgPlotFigures.ax3_x1_y3(ax_arr, x, xlabel, y_arr, ylabel_arr)
        UlgPlotFigures.ax3_x1_y6(ax_arr, x, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-1, 1])
        # ax1.set_ylim([-1, 1])
        # ax2.set_ylim([-1, 1])
        ax_arr[2].legend(['pos', 'vel'])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def actuator_controls_0_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: actuator_controls_0_0')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['control[0]', 'control[1]', 'control[2]', 'control[3]']
        UlgPlotFigures.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        ax_arr[0].set_ylim([-0.1, 0.1])
        ax_arr[1].set_ylim([-0.1, 0.1])
        ax_arr[2].set_ylim([-0.1, 0.1])
        ax_arr[3].set_ylim([0, 0.6])
        # ax1.tick_params(axis=u'y', which=u'both', length=0)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def actuator_outputs_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(1, 1)
        fig.suptitle('Timeseries: actuator_outputs_0')

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2, y3, y4, y5, y6, y7]
        ylabel = 'actuator_outputs_0'
        UlgPlotFigures.ax1_x1_y8(ax_arr, x, xlabel, y_arr, ylabel)
        # ax.set_ylim([700, 2200])

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname + "_a")
        UlgPlotFigures.savefig(jpgfilename, closefig)

        # Next figure

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: actuator_outputs_0')
        xlabel = 'timestamp s'

        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['m1', 'm2', 'm3', 'm4']
        UlgPlotFigures.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([700, 2200])
        # ax1.set_ylim([700, 2200])
        # ax2.set_ylim([700, 2200])
        # ax3.set_ylim([700, 2200])

        y_arr = [y5, y4, y7, y6]
        ylabel_arr = ['m1, m6', 'm2, m5', 'm3, m8', 'm4, m7']
        UlgPlotFigures.ax4_x1_y4(ax_arr, x, xlabel, y_arr, ylabel_arr)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname + "_b")
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def nwindow_hover_pos(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)

        # 1 sec = 10**6 microsec
        # Actual SPS in log files is approx 10
        window = 10*3
        lenx = len(x)
        if window > lenx:
            window = lenx
            print('[nwindow_hover_pos] window %s < len(x) %s ' %
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

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(4, 1)
        arg = 'Timeseries: window = %s, min(std) = %s, time[min(std)] = %s' % \
              (window, round(min_y3_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel = 'timestamp s'
        y_arr = [y0, y1, y2]
        ylabel = 'x, y, z'
        UlgPlotFigures.ax1_x1_y3([ax_arr[0]], x, xlabel, y_arr, ylabel)

        xlabel = 'timestamp s'
        y_arr = [y0_window, y1_window, y2_window, y3_window]
        ylabel = 'std window'
        UlgPlotFigures.ax1_x1_y4([ax_arr[1]], x_window, xlabel, y_arr, ylabel)

        i0 = argmin_y3_window
        il = argmin_y3_window + window

        xlabel = 'timestamp s'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il]]
        ylabel = 'x, y, z'
        UlgPlotFigures.ax1_x1_y3([ax_arr[2]], x[i0:il], xlabel, y_arr, ylabel)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)
        xlabel = 'timestamp s'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel = 'actuator_outputs_0'
        UlgPlotFigures.ax1_x1_y8([ax_arr[3]], x[i0:il], xlabel, y_arr, ylabel)

        csvname = 'hover_nwindow_pos'
        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

    def nwindow_hover_vel(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)

        # 1 sec = 10**6 microsec
        # Actual SPS in log files is approx 10
        window = 10*2
        lenx = len(x)
        if window > lenx:
            window = lenx
            print('[nwindow_hover_vel] window %s < len(x) %s ' %
                  (window, lenx))
            raise RuntimeError

        nmax = len(x) - 1
        ilast = nmax - window + 1
        x_window = x[0:ilast+1]
        fcost = UlgPlotBasics.nwindow_fcost
        y3_window = TimeseriesStats.apply_to_window(y3, fcost, window)
        y4_window = TimeseriesStats.apply_to_window(y4, fcost, window)
        y5_window = TimeseriesStats.apply_to_window(y5, fcost, window)
        y6_window = np.add(y3_window, y4_window, y5_window)

        argmin_y6_window = int(np.argmin(y6_window))
        min_y6_window = y6_window[argmin_y6_window]
        min_x = x[argmin_y6_window]

        [fig, ax_arr] = UlgPlotFigures.create_fig_axes(4, 1)
        arg = 'Timeseries: window = %s, min(fcost) = %s, time[min(fcost)] = %s'\
              % (window, round(min_y6_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel = 'timestamp s'
        y_arr = [y3, y4, y5]
        ylabel = 'vx, vy, vz'
        UlgPlotFigures.ax1_x1_y3([ax_arr[0]], x, xlabel, y_arr, ylabel)

        xlabel = 'timestamp s'
        y_arr = [y3_window, y4_window, y5_window, y6_window]
        ylabel = 'fcost'
        UlgPlotFigures.ax1_x1_y4([ax_arr[1]], x_window, xlabel, y_arr, ylabel)

        i0 = argmin_y6_window
        il = argmin_y6_window + window

        xlabel = 'timestamp s'
        y_arr = [y3[i0:il], y4[i0:il], y5[i0:il]]
        ylabel = 'vx, vy, vz'
        UlgPlotFigures.ax1_x1_y3([ax_arr[2]], x[i0:il], xlabel, y_arr, ylabel)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)

        xlabel = 'timestamp s'
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel = 'actuator variables'
        UlgPlotFigures.ax1_x1_y8([ax_arr[3]], x[i0:il], xlabel, y_arr, ylabel)
        txt = 'std green %s' % round(float(np.std(y0[i0:il])), 2)
        ax_arr[3].annotate(txt, xy=(0.05, 0.8), xycoords='axes fraction')

        csvname = 'hover_nwindow_vel'
        jpgfilename = UlgPlotFigures.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        UlgPlotFigures.savefig(jpgfilename, closefig)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()


if __name__ == '__main__':
    pass
