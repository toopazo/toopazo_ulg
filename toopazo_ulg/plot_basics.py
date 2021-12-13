#!/usr/bin/env python

import os.path
import numpy as np
# from datetime import datetime   # , date, time
import datetime
import matplotlib.pyplot as plt

from toopazo_tools.file_folder import FileFolderTools
from toopazo_tools.time_series import TimeseriesTools
from toopazo_tools.matplotlib import PlotTools, FigureTools

# Check if this is running inside toopazo_ulg/ or deployed as a module
if os.path.isfile('parse_file.py'):
    from parse_file import UlgParser
else:
    from toopazo_ulg.parse_file import UlgParser


class UlgPlotBasics:
    def __init__(self, logdir, tmpdir, plotdir):
        self.logdir = logdir
        self.tmpdir = tmpdir
        self.plotdir = plotdir

        self.figsize = (10, 6)

    @staticmethod
    def save_fig(fig, jpgfilename):
        print(f'[fig_save] Saving {jpgfilename}')
        fig.savefig(jpgfilename)

    @staticmethod
    def get_jpgfilename(pltdir, ulgfile, pltname):
        ulgfile = FileFolderTools.get_file_basename(ulgfile)
        filename = ulgfile.replace('.ulg', '_') + pltname + '.jpg'
        filename = pltdir + '/' + filename
        return filename

    @staticmethod
    def timestamp_to_datetime(x):
        xdt = []
        for tstamp in x:
            xdt.append(datetime.datetime.fromtimestamp(tstamp))
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

        [fig, ax_arr] = FigureTools.create_fig_axes(3, 1)
        fig.suptitle('Timeseries: vehicle_attitude_0_deg')

        xlabel = 'timestamp s'
        x_arr = [x]
        y_arr = [y0, y1, y2]
        ylabel_arr = ['Roll deg', 'Pitch deg', 'Yaw deg']
        PlotTools.ax3_x1_y3(ax_arr, x_arr, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-30, 30])
        # ax1.set_ylim([-30, 30])
        # ax2.set_ylim([-30, 30])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def vehicle_rates_setpoint_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5] = \
            UlgParser.get_vehicle_rates_setpoint_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)
        _ = [y3, y4]

        [fig, ax_arr] = FigureTools.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: vehicle_rates_setpoint_0')

        xlabel = 'timestamp s'
        x_arr = [x]
        y_arr = [y0, y1, y2, y5]
        ylabel_arr = ['roll', 'pitch', 'yaw', 'thrust_body']
        PlotTools.ax4_x1_y4(ax_arr, x_arr, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-90, 90])
        # ax1.set_ylim([-90, 90])
        # ax2.set_ylim([-90, 90])
        # ax3.set_ylim([0, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def manual_control_setpoint_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_manual_control_setpoint_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        # fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        # ax_arr = [ax0, ax1, ax2]
        # fig = plt.figure()
        # ax = plt.gca()
        [fig, ax_arr] = FigureTools.create_fig_axes(1, 1)
        fig.suptitle('Timeseries: manual_control_setpoint_0')

        xlabel = 'timestamp s'
        x_arr = [x]
        y_arr = [y0, y1, y2, y3]
        ylabel = 'RC inputs'
        PlotTools.ax1_x1_y4(ax_arr, x_arr, xlabel, y_arr, ylabel)
        # ax.set_ylim([-1, 1])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def vehicle_local_position_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)
        _ = [y6, y7, y8]

        [fig, ax_arr] = FigureTools.create_fig_axes(3, 1)
        fig.suptitle('Timeseries: vehicle_local_position_0')

        xlabel = 'timestamp s'
        x_arr = [x]
        y_arr = [y0, y1, y2, y3, y4, y5]
        ylabel_arr = ['x m, vx m/s', 'y m, vy m/s', 'z m, vz m/s']
        PlotTools.ax3_x1_y6(ax_arr, x_arr, xlabel, y_arr, ylabel_arr)
        # ax0.set_ylim([-1, 1])
        # ax1.set_ylim([-1, 1])
        # ax2.set_ylim([-1, 1])
        ax_arr[2].legend(['pos', 'vel'])

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def actuator_controls_0_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = FigureTools.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: actuator_controls_0_0')

        xlabel_arr = ['timestamp s']
        x_arr = [x]
        y_arr = [y0, y1, y2, y3]
        ylabel_arr = ['control[0]', 'control[1]', 'control[2]', 'control[3]']
        PlotTools.ax4_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr)
        ax_arr[0].set_ylim([-0.1, 0.1])
        ax_arr[1].set_ylim([-0.1, 0.1])
        ax_arr[2].set_ylim([-0.1, 0.1])
        ax_arr[3].set_ylim([0, 0.6])
        # ax1.tick_params(axis=u'y', which=u'both', length=0)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def actuator_outputs_0(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)
        # x = UlgPlotMixer.timestamp_to_datetime(x)

        [fig, ax_arr] = FigureTools.create_fig_axes(1, 1)
        fig.suptitle('Timeseries: actuator_outputs_0')

        xlabel_arr = ['timestamp s']
        x_arr = [x]
        y_arr = [y0, y1, y2, y3, y4, y5, y6, y7]
        ylabel = 'actuator_outputs_0'
        PlotTools.ax1_x1_y8(ax_arr, x_arr, xlabel_arr, y_arr, ylabel)
        # ax.set_ylim([700, 2200])

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname + "_a")
        FigureTools.savefig(jpgfilename, closefig)

        # Next figure

        [fig, ax_arr] = FigureTools.create_fig_axes(4, 1)
        fig.suptitle('Timeseries: actuator_outputs_0')
        xlabel_arr = ['timestamp s']

        # y_arr = [y0, y1, y2, y3]
        # x_arr = [x]
        # ylabel_arr = ['m1', 'm2', 'm3', 'm4']
        # PlotTools.ax4_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr)
        # # ax0.set_ylim([700, 2200])
        # # ax1.set_ylim([700, 2200])
        # # ax2.set_ylim([700, 2200])
        # # ax3.set_ylim([700, 2200])

        y_arr = [y0, y1, y2, y3, y5, y4, y7, y6]
        x_arr = [x]
        ylabel_arr = ['m1, m6', 'm2, m5', 'm3, m8', 'm4, m7']
        PlotTools.ax4_x1_y8(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname + "_b")
        FigureTools.savefig(jpgfilename, closefig)

    def nwindow_hover_pos(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y3, y4, y5, y6, y7, y8]

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
        y0_window = TimeseriesTools.apply_to_window(y0, np.std, window)
        y1_window = TimeseriesTools.apply_to_window(y1, np.std, window)
        y2_window = TimeseriesTools.apply_to_window(y2, np.std, window)
        y3_window = np.add(y0_window, y1_window, y2_window)

        argmin_y3_window = int(np.argmin(y3_window))
        min_y3_window = y3_window[argmin_y3_window]
        min_x = x[argmin_y3_window]

        [fig, ax_arr] = FigureTools.create_fig_axes(4, 1)
        arg = 'Timeseries: window = %s, min(std) = %s, time[min(std)] = %s' % \
              (window, round(min_y3_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel_arr = ['timestamp s']
        x_arr = [x]
        y_arr = [y0, y1, y2]
        ylabel_arr = ['x, y, z']
        PlotTools.ax1_x1_y3([ax_arr[0]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        xlabel_arr = ['timestamp s']
        x_arr = [x_window]
        y_arr = [y0_window, y1_window, y2_window, y3_window]
        ylabel_arr = ['std window']
        PlotTools.ax1_x1_y4([ax_arr[1]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        i0 = argmin_y3_window
        il = argmin_y3_window + window

        xlabel_arr = ['timestamp s']
        x_arr = [x[i0:il]]
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il]]
        ylabel_arr = ['x, y, z']
        PlotTools.ax1_x1_y3([ax_arr[2]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)
        _ = csvname

        xlabel_arr = ['timestamp s']
        x_arr = [x[i0:il]]
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel_arr = ['actuator_outputs_0']
        PlotTools.ax1_x1_y8([ax_arr[3]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        csvname = 'hover_nwindow_pos'
        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def nwindow_hover_vel(self, ulgfile, closefig):
        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2, y6, y7, y8]

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
        y3_window = TimeseriesTools.apply_to_window(y3, fcost, window)
        y4_window = TimeseriesTools.apply_to_window(y4, fcost, window)
        y5_window = TimeseriesTools.apply_to_window(y5, fcost, window)
        y6_window = np.add(y3_window, y4_window, y5_window)

        argmin_y6_window = int(np.argmin(y6_window))
        min_y6_window = y6_window[argmin_y6_window]
        min_x = x[argmin_y6_window]

        [fig, ax_arr] = FigureTools.create_fig_axes(4, 1)
        arg = 'Timeseries: window = %s, min(fcost) = %s, time[min(fcost)] = %s'\
              % (window, round(min_y6_window, 2), round(min_x, 2))
        fig.suptitle(arg)

        xlabel_arr = ['timestamp s']
        x_arr = [x]
        y_arr = [y3, y4, y5]
        ylabel_arr = ['vx, vy, vz']
        PlotTools.ax1_x1_y3([ax_arr[0]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        xlabel_arr = ['timestamp s']
        x_arr = [x_window]
        y_arr = [y3_window, y4_window, y5_window, y6_window]
        ylabel_arr = ['fcost']
        PlotTools.ax1_x1_y4([ax_arr[1]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        i0 = argmin_y6_window
        il = argmin_y6_window + window

        xlabel_arr = ['timestamp s']
        x_arr = [x[i0:il]]
        y_arr = [y3[i0:il], y4[i0:il], y5[i0:il]]
        ylabel_arr = ['vx, vy, vz']
        PlotTools.ax1_x1_y3([ax_arr[2]], x_arr, xlabel_arr, y_arr, ylabel_arr)

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7] = \
            UlgParser.get_actuator_outputs_0(ulgfile, self.tmpdir)
        _ = csvname

        xlabel_arr = ['timestamp s']
        x_arr = [x[i0:il]]
        y_arr = [y0[i0:il], y1[i0:il], y2[i0:il], y3[i0:il],
                 y4[i0:il], y5[i0:il], y6[i0:il], y7[i0:il]]
        ylabel_arr = ['actuator variables']
        PlotTools.ax1_x1_y8([ax_arr[3]], x_arr, xlabel_arr, y_arr, ylabel_arr)
        txt = 'std green %s' % round(float(np.std(y0[i0:il])), 2)
        ax_arr[3].annotate(txt, xy=(0.05, 0.8), xycoords='axes fraction')

        csvname = 'hover_nwindow_vel'
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

    def pos_vel(self, ulgfile, time_win):
        df_pv = UlgParser.get_pandas_dataframe_pos_vel(self.tmpdir, ulgfile, time_win)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        df_pv.plot(y=['x', 'y', 'z', 'pnorm'], ax=ax1, grid=True)
        ax1.set_ylabel("Position m")
        ax1.set_xlabel("Time s")
        df_pv.plot(y=['vx', 'vy', 'vz', 'vnorm'], ax=ax2, grid=True)
        ax2.set_ylabel("Velocity m/s")
        ax2.set_xlabel("Time s")

        plt.subplots_adjust(wspace=0, hspace=0.1)
        pltname = self.pos_vel.__name__
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, pltname)
        UlgPlotBasics.save_fig(fig, jpgfilename)

    def rpy_angles(self, ulgfile, time_win):
        [df_att, df_attsp] = UlgParser.get_pandas_dataframe_rpy_angles(self.tmpdir, ulgfile, time_win)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        df_att.plot(y=['roll'], ax=ax1, grid=True)
        df_attsp.plot(y=['roll sp'], ax=ax1, grid=True)
        ax1.set_ylabel("Roll deg")
        ax1.set_xlabel("Time s")
        df_att.plot(y=['pitch'], ax=ax2, grid=True)
        df_attsp.plot(y=['pitch sp'], ax=ax2, grid=True)
        ax2.set_ylabel("Pitch deg")
        ax2.set_xlabel("Time s")
        df_att.plot(y=['yaw'], ax=ax3, grid=True)
        df_attsp.plot(y=['yaw sp'], ax=ax3, grid=True)
        ax3.set_ylabel("Yaw deg")
        ax3.set_xlabel("Time s")

        plt.subplots_adjust(wspace=0, hspace=0.1)
        pltname = self.rpy_angles.__name__
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, pltname)
        UlgPlotBasics.save_fig(fig, jpgfilename)

    def pqr_angvel(self, ulgfile, time_win):
        [df_angvel, df_angvelsp] = UlgParser.get_pandas_dataframe_pqr_angvel(self.tmpdir, ulgfile, time_win)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        df_angvel.plot(y=['roll rate'], ax=ax1, grid=True)
        df_angvelsp.plot(y=['roll rate sp'], ax=ax1, grid=True)
        ax1.set_ylabel("Roll rate deg/s")
        ax1.set_xlabel("Time s")
        df_angvel.plot(y=['pitch rate'], ax=ax2, grid=True)
        df_angvelsp.plot(y=['pitch rate sp'], ax=ax2, grid=True)
        ax2.set_ylabel("Pitch rate deg/s")
        ax2.set_xlabel("Time s")
        df_angvel.plot(y=['yaw rate'], ax=ax3, grid=True)
        df_angvelsp.plot(y=['yaw rate sp'], ax=ax3, grid=True)
        ax3.set_ylabel("Yaw rate deg/s")
        ax3.set_xlabel("Time s")

        plt.subplots_adjust(wspace=0, hspace=0.1)
        pltname = self.pqr_angvel.__name__
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, pltname)
        UlgPlotBasics.save_fig(fig, jpgfilename)

    def man_ctrl(self, ulgfile, time_win):
        [df_sticks, df_switches] = UlgParser.get_pandas_dataframe_man_ctrl(self.tmpdir, ulgfile, time_win)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        df_sticks.plot(y=['roll stick', 'pitch stick', 'yaw stick', 'throttle stick'], ax=ax1, grid=True)
        ax1.set_ylabel("Stick displacement")
        ax1.set_xlabel("Time s")
        df_switches.plot(y=['arm_switch', 'offboard_switch', 'kill_switch', 'mode_switch'], ax=ax2, grid=True)
        ax2.set_ylabel("Switch state")
        ax2.set_xlabel("Time s")

        plt.subplots_adjust(wspace=0, hspace=0.1)
        pltname = self.man_ctrl.__name__
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, pltname)
        UlgPlotBasics.save_fig(fig, jpgfilename)

    def ctrl_alloc(self, ulgfile, time_win):
        [df_in, df_out] = UlgParser.get_pandas_dataframe_ctrl_alloc(self.tmpdir, ulgfile, time_win)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        df_in.plot(y=['roll rate cmd', 'pitch rate cmd', 'yaw rate cmd', 'az cmd'], ax=ax1, grid=True)
        ax1.set_ylabel("Cmd accelerations (input)")
        ax1.set_xlabel("Time s")
        df_out.plot(y=['output[0]', 'output[1]', 'output[2]', 'output[3]',
                       'output[4]', 'output[5]', 'output[6]', 'output[7]'], ax=ax2, grid=True)
        ax2.set_ylabel("Cmd actuator (output)")
        ax2.set_xlabel("Time s")

        plt.subplots_adjust(wspace=0, hspace=0.1)
        pltname = self.ctrl_alloc.__name__
        jpgfilename = self.get_jpgfilename(self.plotdir, ulgfile, pltname)
        UlgPlotBasics.save_fig(fig, jpgfilename)


if __name__ == '__main__':
    pass
