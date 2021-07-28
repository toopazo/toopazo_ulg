#!/usr/bin/env python

# from toopazo_tools.fileFolderTools import FileFolderTools
# from toopazo_tools.statistics import TimeseriesStats
# from toopazo_tools.matplotlibTools import PlotTools
from toopazo_tools.matplotlib import FigureTools

from toopazo_ulg.file_parser import UlgParser
from toopazo_ulg.plot_basics import UlgPlotBasics

import numpy as np


class UlgPlotSysid(UlgPlotBasics):
    def cmd_roll_to_attitude(self, ulgfile, closefig):
        [fig, ax_arr] = FigureTools.create_fig_axes(5, 1)
        fig.suptitle('sysID: closed loop, stabilized mode')

        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        _ = [csvname, y1, y2, y3]

        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y0)
        ax_arr[0].set(xlabel='Time s', ylabel='cmd roll')
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)
        ax_arr[0].set_ylim([-0.2, 0.2])

        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, self.tmpdir)
        _ = csvname

        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y0)
        ax_arr[1].set(xlabel='Time s', ylabel='roll deg')
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)
        ax_arr[1].set_ylim([-20, 20])

        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y1)
        ax_arr[2].set(xlabel='Time s', ylabel='pitch deg')
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)
        ax_arr[2].set_ylim([-20, 20])

        ax_arr[3].grid(True)
        ax_arr[3].plot(x, y2-np.mean(y2))
        ax_arr[3].set(xlabel='Time s', ylabel='yaw deg')
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)
        ax_arr[3].set_ylim([-20, 20])

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2, y3, y4, y5, y6, y7]

        ax_arr[4].grid(True)
        ax_arr[4].plot(x, y8)
        ax_arr[4].set(xlabel='Time s', ylabel='az m/s2')
        ax_arr[4].locator_params(axis='y', nbins=3)
        ax_arr[4].locator_params(axis='x', nbins=4)
        ax_arr[4].set_ylim([-5, 5])

        csvname = "sysid_roll"
        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def cmd_pitch_to_attitude(self, ulgfile, closefig):
        [fig, ax_arr] = FigureTools.create_fig_axes(5, 1)
        fig.suptitle('sysID: closed loop, stabilized mode')

        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y2, y3]

        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y1)
        ax_arr[0].set(xlabel='Time s', ylabel='cmd pitch')
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)
        ax_arr[0].set_ylim([-0.2, 0.2])

        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, self.tmpdir)
        _ = csvname

        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y0)
        ax_arr[1].set(xlabel='Time s', ylabel='roll deg')
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)
        ax_arr[1].set_ylim([-20, 20])

        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y1)
        ax_arr[2].set(xlabel='Time s', ylabel='pitch deg')
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)
        ax_arr[2].set_ylim([-20, 20])

        ax_arr[3].grid(True)
        ax_arr[3].plot(x, y2-np.mean(y2))
        ax_arr[3].set(xlabel='Time s', ylabel='yaw deg')
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)
        ax_arr[3].set_ylim([-20, 20])

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2, y3, y4, y5, y6, y7]

        ax_arr[4].grid(True)
        ax_arr[4].plot(x, y8)
        ax_arr[4].set(xlabel='Time s', ylabel='az m/s2')
        ax_arr[4].locator_params(axis='y', nbins=3)
        ax_arr[4].locator_params(axis='x', nbins=4)
        ax_arr[4].set_ylim([-5, 5])

        csvname = "sysid_pitch"
        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def cmd_yawrate_to_attitude(self, ulgfile, closefig):
        [fig, ax_arr] = FigureTools.create_fig_axes(5, 1)
        fig.suptitle('sysID: closed loop, stabilized mode')

        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y3]

        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y2)
        ax_arr[0].set(xlabel='Time s', ylabel='cmd yawrate')
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)
        ax_arr[0].set_ylim([-0.5, 0.5])

        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, self.tmpdir)
        _ = csvname

        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y0)
        ax_arr[1].set(xlabel='Time s', ylabel='roll deg')
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)
        ax_arr[1].set_ylim([-20, 20])

        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y1)
        ax_arr[2].set(xlabel='Time s', ylabel='pitch deg')
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)
        ax_arr[2].set_ylim([-20, 20])

        ax_arr[3].grid(True)
        ax_arr[3].plot(x, y2-np.mean(y2))
        ax_arr[3].set(xlabel='Time s', ylabel='yaw deg')
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)
        ax_arr[3].set_ylim([-120, 120])

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2, y3, y4, y5, y6, y7]

        ax_arr[4].grid(True)
        ax_arr[4].plot(x, y8)
        ax_arr[4].set(xlabel='Time s', ylabel='az m/s2')
        ax_arr[4].locator_params(axis='y', nbins=3)
        ax_arr[4].locator_params(axis='x', nbins=4)
        ax_arr[4].set_ylim([-5, 5])

        csvname = "sysid_yawrate"
        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)

    def cmd_az_to_attitude(self, ulgfile, closefig):
        [fig, ax_arr] = FigureTools.create_fig_axes(5, 1)
        fig.suptitle('sysID: closed loop, stabilized mode')

        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_actuator_controls_0_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2]

        ax_arr[0].grid(True)
        ax_arr[0].plot(x, y3)
        ax_arr[0].set(xlabel='Time s', ylabel='cmd az')
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)
        ax_arr[0].set_ylim([0, 0.8])

        [csvname, x, y0, y1, y2] = \
            UlgParser.get_vehicle_attitude_0_deg(ulgfile, self.tmpdir)
        _ = csvname

        ax_arr[1].grid(True)
        ax_arr[1].plot(x, y0)
        ax_arr[1].set(xlabel='Time s', ylabel='roll deg')
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)
        ax_arr[1].set_ylim([-20, 20])

        ax_arr[2].grid(True)
        ax_arr[2].plot(x, y1)
        ax_arr[2].set(xlabel='Time s', ylabel='pitch deg')
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)
        ax_arr[2].set_ylim([-20, 20])

        ax_arr[3].grid(True)
        ax_arr[3].plot(x, y2-np.mean(y2))
        ax_arr[3].set(xlabel='Time s', ylabel='yaw deg')
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)
        ax_arr[3].set_ylim([-20, 20])

        [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8] = \
            UlgParser.get_vehicle_local_position_0(ulgfile, self.tmpdir)
        _ = [csvname, y0, y1, y2, y3, y4, y5, y6, y7]

        ax_arr[4].grid(True)
        ax_arr[4].plot(x, y8)
        ax_arr[4].set(xlabel='Time s', ylabel='az m/s2')
        ax_arr[4].locator_params(axis='y', nbins=3)
        ax_arr[4].locator_params(axis='x', nbins=4)
        ax_arr[4].set_ylim([-10, 10])

        csvname = "sysid_az"
        jpgfilename = self.get_jpgfilename(
            self.plotdir, ulgfile, csvname)
        FigureTools.savefig(jpgfilename, closefig)


if __name__ == '__main__':
    pass
