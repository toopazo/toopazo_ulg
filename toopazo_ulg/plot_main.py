#!/usr/bin/env python3

import os
import argparse
# import sys
# import subprocess
# import csv
# import matplotlib.pyplot as plt

from toopazo_tools.file_folder import FileFolderTools as FFTools
# from toopazo_tools.statistics import TimeseriesStats

# Check if this is running inside toopazo_ulg/ or deployed as a module
if os.path.isfile('parse_file.py'):
    from parse_file import UlgParser
    from plot_basics import UlgPlotBasics
    from plot_sysid import UlgPlotSysid
    from plot_mixer import UlgPlotMixer
else:
    from toopazo_ulg.parse_file import UlgParser
    from toopazo_ulg.plot_basics import UlgPlotBasics
    from toopazo_ulg.plot_sysid import UlgPlotSysid
    from toopazo_ulg.plot_mixer import UlgPlotMixer


class UlgPlot:
    def __init__(self, bdir, time_win,
                 pos_vel, rpy_angles, pqr_angvel, man_ctrl, ctrl_alloc):
        # bdir = FFTools.full_path(args.bdir)
        # bdir = os.path.abspath(args.bdir)
        self.logdir = bdir + '/logs'
        self.tmpdir = bdir + '/tmp'
        self.plotdir = bdir + '/plots'
        try:
            if not os.path.isdir(self.logdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.tmpdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.plotdir):
                os.mkdir(self.logdir)
        except OSError:
            raise RuntimeError('Directories are not present or could not be created')

        self.file_extension = 'ulg'
        self.time_win = time_win

        self.pos_vel = pos_vel
        self.rpy_angles = rpy_angles
        self.pqr_angvel = pqr_angvel
        self.man_ctrl = man_ctrl
        self.ctrl_alloc = ctrl_alloc

        self.ulg_plot_basics = UlgPlotBasics(
            self.logdir, self.tmpdir, self.plotdir)

        self.ulg_plot_sysid = UlgPlotSysid(
            self.logdir, self.tmpdir, self.plotdir)

        self.ulg_plot_mixer = UlgPlotMixer(
            self.logdir, self.tmpdir, self.plotdir)

        # Remove all files from tmpdir
        # self.ulgparser.clear_tmpdir()

    def process_file(self, ulg_file):
        print('[process_file] Working on file %s' % ulg_file)

        UlgParser.check_ulog2csv(self.tmpdir, ulg_file)

        if self.pos_vel:
            self.ulg_plot_basics.pos_vel(ulg_file, self.time_win)

        if self.rpy_angles:
            self.ulg_plot_basics.rpy_angles(ulg_file, self.time_win)

        if self.pqr_angvel:
            self.ulg_plot_basics.pqr_angvel(ulg_file, self.time_win)

        if self.man_ctrl:
            self.ulg_plot_basics.man_ctrl(ulg_file, self.time_win)

        if self.ctrl_alloc:
            self.ulg_plot_basics.ctrl_alloc(ulg_file, self.time_win)
            self.ulg_plot_mixer.ctrl_alloc_model(ulg_file, self.time_win)
            # self.ulg_plot_mixer.mixer_input_output(ulg_file, closefig)
            # self.ulg_plot_basics.actuator_controls_0_0(ulg_file, closefig)
            # self.ulg_plot_basics.actuator_outputs_0(ulg_file, closefig)

        # self.ulg_plot_basics.nwindow_hover_pos(ulg_file, closefig)
        # self.ulg_plot_basics.nwindow_hover_vel(ulg_file, closefig)

        # self.ulg_plot_sysid.cmd_roll_to_attitude(ulg_file, closefig)
        # self.ulg_plot_sysid.cmd_pitch_to_attitude(ulg_file, closefig)
        # self.ulg_plot_sysid.cmd_yawrate_to_attitude(ulg_file, closefig)
        # self.ulg_plot_sysid.cmd_az_to_attitude(ulg_file, closefig)

    def process_folder(self):
        print('[process_folder] Working on folder %s' % self.logdir)
        # foldername, extension, method
        FFTools.run_method_on_folder(self.logdir, self.file_extension, self.process_file)
        
    def find_ulg_file(self, log_num):
        ulg_file = 'No such file'
        file_arr = FFTools.get_file_arr(fpath=self.logdir, extension=self.file_extension)
        for file in file_arr:
            if log_num is not None:
                pattern = f'_{log_num}_'
                if pattern in file:
                    ulg_file = os.path.abspath(file)
                    break
        print(f'[find_ulg_file] logdir {self.logdir}')
        print(f'[find_ulg_file] file_extension {self.file_extension}')
        print(f'[find_ulg_file] log_num {log_num}')
        print(f'[find_ulg_file] ulg_file {ulg_file}')
        return ulg_file


print('[plot_main] module name is %s' % __name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .ulg files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--log_num', action='store', default=None,
                        help='Specific log number to process')
    parser.add_argument('--time_win', action='store', default=None,
                        help='Specific time window to process', nargs=2, type=float)
    # parser.add_argument('--plot', action='store_true', required=False,
    #                     help='plot results')
    parser.add_argument('--pos_vel', action='store_true', help='pos and vel')
    parser.add_argument('--rpy_angles', action='store_true', help='roll, pitch and yaw attitude angles')
    parser.add_argument('--pqr_angvel', action='store_true', help='P, Q, and R body angular velocity')
    parser.add_argument('--man_ctrl', action='store_true', help='manual control')
    parser.add_argument('--ctrl_alloc', action='store_true', help='Control allocation and in/out analysis')

    args = parser.parse_args()

    ulg_plot = UlgPlot(
        os.path.abspath(args.bdir), args.time_win,
        args.pos_vel, args.rpy_angles, args.pqr_angvel, args.man_ctrl, args.ctrl_alloc
    )
    if args.log_num is not None:
        ulg_plot.process_file(ulg_file=ulg_plot.find_ulg_file(args.log_num))
    else:
        ulg_plot.process_folder()

    # Run it as a package
    # python -m toopazo_ulg.plot_main \
    # --bdir . --time_win 2240 2349 --log_num 43 \
    # --pos_vel --rpy_angles --pqr_angvel --man_ctrl --ctrl_alloc
