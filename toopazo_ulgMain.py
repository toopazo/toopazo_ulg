#!/usr/bin/env python

import sys
import argparse
import subprocess
import csv
import matplotlib.pyplot as plt

from toopazo_statistics import TimeseriesStats
from toopazo_utilities import FileFolderUtils
from toopazo_ulgParser import UlgParser
from toopazo_ulgPlotter import UlgPlotter


class UlgProcess:
    def __init__(self, logdir, tmpdir, plotdir):
        self.logdir = logdir
        self.tmpdir = tmpdir
        self.plotdir = plotdir

        self.ulgplotter = UlgPlotter(self.plotdir)

        # Remove all files from tmpdir
        # self.ulgparser.clear_tmpdir()

    def check_ulog2csv(self, ulgfile):

        # Check if we need to run ulog2csv
        csvname = 'actuator_controls_0_0'
        csvfile = UlgParser.get_csvfile(self.tmpdir, ulgfile, csvname)
        if FileFolderUtils.is_file(csvfile):
            UlgParser.ulog2info(ulgfile)
        else:
            UlgParser.ulog2csv(ulgfile, self.tmpdir)
            UlgParser.write_vehicle_attitude_0_deg(ulgfile, self.tmpdir)

    def process_file(self, ulgfile):

        print('[process_file] processing %s' % ulgfile)

        self.check_ulog2csv(ulgfile)

        self.ulgplotter.plot_actuator_controls_0_0(ulgfile, self.tmpdir)
        self.ulgplotter.plot_actuator_outputs_0(ulgfile, self.tmpdir)
        # self.ulgplotter.plot_vehicle_local_position_0(ulgfile, self.tmpdir)
        # self.ulgplotter.plot_manual_control_setpoint_0(ulgfile, self.tmpdir)
        # self.ulgplotter.plot_vehicle_rates_setpoint_0(ulgfile, self.tmpdir)
        self.ulgplotter.plot_vehicle_attitude_0_deg(ulgfile, self.tmpdir)

        # self.ulgplotter.plot_nwindow_hover_pos()
        self.ulgplotter.plot_nwindow_hover_vel(ulgfile, self.tmpdir)

        plt.show()

    def process_logdir(self):
        print('[process_logdir] processing %s' % self.logdir)

        # foldername, extension, method
        FileFolderUtils.run_method_on_folder(ulogdir, '.ulg',
                                             ulgprocess.process_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse and process ulg files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders ')
    # parser.add_argument('--plot', action='store_true', required=False,
    #                     help='plot results')
    # parser.add_argument('--loc', action='store_true', help='location')
    args = parser.parse_args()

    bdir = FileFolderUtils.full_path(args.bdir)
    # uplot = args.plot

    ulogdir = bdir + '/logs'
    utmpdir = bdir + '/tmp'
    uplotdir = bdir + '/plots'

    ulgprocess = UlgProcess(ulogdir, utmpdir, uplotdir)
    ulgprocess.process_logdir()

    # python /home/tzo4/Dropbox/tomas/pennState/avia/software/ulgPlotter/toopazo_ulgMain.py --bdir .
    # python toopazo_ulgMain.py --file logs/log_49_2019-1-16-13-22-24.ulg
