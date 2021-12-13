#!/usr/bin/env python

from toopazo_tools.file_folder import FileFolderTools
from toopazo_tools.pandas import PandasTools

import subprocess
import csv
import numpy as np
import math
import pandas as pd


class UlgParser:
    @staticmethod
    def clear_tmpdir(tmpdir):
        # remove all from tmpdir
        FileFolderTools.clear_folders(tmpdir)

    @staticmethod
    def ulog2csv(ulgfile, tmpdir):
        cmd_line = 'ulog2csv -o %s %s' % (tmpdir, ulgfile)
        print(cmd_line)
        try:
            byte_string = subprocess.check_output(
                cmd_line, stderr=subprocess.STDOUT, shell=True)
            print(byte_string.decode("utf-8"))
        except subprocess.CalledProcessError:
            print('[ulog2info] Error processing %s' % ulgfile)
            return False
        else:
            return True

    @staticmethod
    def write_vehicle_attitude_0_deg(ulgfile, tmpdir):
        assert isinstance(ulgfile, str)

        [csvname, x, y0, y1, y2, y3] = \
            UlgParser.get_vehicle_attitude_0(ulgfile, tmpdir)
        [y0, y1, y2] = UlgParser.quat2rpy([y0, y1, y2, y3])
        _ = csvname

        # UlgParser.parse_csv() converts timestamp field from microseconds
        # to seconds. So, we need to multiply x = csvd['timestamp'] by 10**6
        # to write in microseconds again
        # https://dev.px4.io/v1.9.0/en/log/ulog_file_format.html
        x = np.uint64(x * 10**6)

        csvname = "vehicle_attitude_0_deg"
        csvfile = UlgParser.get_csvfile(tmpdir, ulgfile, csvname)
        print("[write_vehicle_attitude_0_deg] ulgfile = %s" % csvfile)

        with open(csvfile, 'w') as csvfd:
            csvwriter = csv.writer(csvfd, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['timestamp', 'roll', 'pitch', 'yaw'])

            # csvwriter.writerow([x, y0, y1, y2])
            line_arr = np.array([x, y0, y1, y2])
            line_arr = np.transpose(line_arr)

            lshape = np.shape(line_arr)
            nrow = lshape[0]
            ncol = lshape[1]
            print("[write_vehicle_attitude_0_deg] nrow = %s" % nrow)
            print("[write_vehicle_attitude_0_deg] ncol = %s" % ncol)

            for i in range(0, nrow):
                q = line_arr[i]
                csvwriter.writerow([q[0], q[1], q[2], q[3]])

    @staticmethod
    def ulog2info(ulgfile):
        cmd_line = 'ulog_info %s' % ulgfile
        print(cmd_line)
        try:
            byte_string = subprocess.check_output(
                cmd_line, stderr=subprocess.STDOUT, shell=True)
            print(byte_string.decode("utf-8"))
        except subprocess.CalledProcessError:
            print('[ulog2info] Error processing %s' % ulgfile)
            return False
        else:
            return True

    @staticmethod
    def get_csvfile(tmpdir, ulgfile, csvname):
        ulgfile = FileFolderTools.get_file_basename(ulgfile)
        csvfile = ulgfile.replace('.ulg', '_') + csvname + '.csv'
        csvfile = tmpdir + '/' + csvfile
        # print('[get_csvfile] csvfile %s' % csvfile)
        return csvfile

    @staticmethod
    def parse_csv(ulgfile, csvname, tmpdir):
        # https://github.com/PX4/Firmware/tree/master/msg

        # log_49_2019-1-16-13-22-24.ulg
        # log_49_2019-1-16-13-22-24_actuator_controls_0_0.csv
        # csvname = 'actuator_controls_0_0'
        csvfile = UlgParser.get_csvfile(tmpdir, ulgfile, csvname)
        # print('[parse_csv] ulgfile %s' %
        #       FileFolderTools.get_file_basename(ulgfile))
        # print('[parse_csv] csvfile %s' %
        #       FileFolderTools.get_file_basename(csvfile))
        csv_fd = open(csvfile)

        reader = csv.DictReader(csv_fd)
        csvd = {}
        cnt = 1
        for row in reader:
            if cnt == 1:
                # Create array with values
                for key, value in row.items():
                    csvd[key] = [value]
                    # arg = '[parse_csv] csvd[%s] = %s' % (key, value)
                    # print(arg)
            else:
                # Append array with values
                for key, value in row.items():
                    csvd[key].append(value)
            # Update counter
            cnt += 1

        # # Print dict
        # for key, value in csvd.items():
        #     arg = '[parse_csv] %s[%s]' % (key, value)
        #     print(arg)

        # Convert all values to float numpy arrays
        for key, value in csvd.items():
            x = np.array(value)
            y = x.astype(np.float)
            csvd[key] = y

        # Convert timestamp to datetime
        for i in range(0, len(csvd['timestamp'])):
            # https://dev.px4.io/v1.9.0/en/log/ulog_file_format.html
            # Timestamp is a uint64_t integer, denotes the start of the
            # logging in microseconds.
            logging_secs = float(csvd['timestamp'][i] / 10**6)
            csvd['timestamp'][i] = logging_secs

        return csvd

    @staticmethod
    def quat2eulerangles(q):
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        # roll(x - axis rotation)
        sinr_cosp = +2.0 * (qw * qx + qy * qz)
        cosr_cosp = +1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch(y - axis rotation)
        sinp = +2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            # use 90 degrees if out of range
            sign_sinp = sinp / abs(sinp)
            pitch = (math.pi / 2) * sign_sinp
        else:
            pitch = math.asin(sinp)

        # yaw(z - axis rotation)
        siny_cosp = +2.0 * (qw * qz + qx * qy)
        cosy_cosp = +1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        rpy = [roll, pitch, yaw]
        return rpy

    @staticmethod
    def quat2rpy(q_arr):
        q_arr = np.array(q_arr)
        q_arr = np.transpose(q_arr)

        qshape = np.shape(q_arr)
        nrow = qshape[0]
        ncol = qshape[1]
        if ncol != 4:
            raise RuntimeError('ncol != 4')

        rad2deg = 180 / math.pi
        r_arr = []
        p_arr = []
        y_arr = []
        for i in range(0, nrow):
            q = q_arr[i]
            rpy = UlgParser.quat2eulerangles(q)
            # print(rpy)
            r_arr.append(rpy[0] * rad2deg)
            p_arr.append(rpy[1] * rad2deg)
            y_arr.append(rpy[2] * rad2deg)
        return [r_arr, p_arr, y_arr]

    @staticmethod
    def get_vehicle_attitude_0(ulgfile, tmpdir):
        csvname = 'vehicle_attitude_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['q[0]']
        y1 = csvd['q[1]']
        y2 = csvd['q[2]']
        y3 = csvd['q[3]']

        return [csvname, x, y0, y1, y2, y3]

    @staticmethod
    def get_vehicle_attitude_0_deg(ulgfile, tmpdir):
        csvname = 'vehicle_attitude_0_deg'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['roll']
        y1 = csvd['pitch']
        y2 = csvd['yaw']

        return [csvname, x, y0, y1, y2]

    @staticmethod
    def get_vehicle_rates_setpoint_0(ulgfile, tmpdir):
        csvname = 'vehicle_rates_setpoint_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['roll']
        y1 = csvd['pitch']
        y2 = csvd['yaw']
        # y3 = csvd['thrust']
        y3 = csvd['thrust_body[0]']
        y4 = csvd['thrust_body[1]']
        y5 = csvd['thrust_body[2]']
        # https://github.com/PX4/PX4-Autopilot/blob/master/msg/
        # vehicle_attitude_setpoint.msg
        #
        # # For clarification: For multicopters thrust_body[0] and thrust[1]
        # are usually 0 and thrust[2] is the negative throttle demand.
        # # For fixed wings thrust_x is the throttle demand and thrust_y,
        # thrust_z will usually be zero.
        # float32[3] thrust_body		# Normalized thrust command in
        # body NED frame [-1,1]

        return [csvname, x, y0, y1, y2, y3, y4, y5]

    @staticmethod
    def get_toopazo_ctrlalloc_0(ulgfile, tmpdir):
        csvname = 'toopazo_ctrlalloc_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']

        status = csvd['status']

        c0 = csvd['controls[0]']
        c1 = csvd['controls[1]']
        c2 = csvd['controls[2]']
        c3 = csvd['controls[3]']
        controls = [c0, c1, c2, c3]

        o0 = csvd['output[0]']
        o1 = csvd['output[1]']
        o2 = csvd['output[2]']
        o3 = csvd['output[3]']
        o4 = csvd['output[4]']
        o5 = csvd['output[5]']
        o6 = csvd['output[6]']
        o7 = csvd['output[7]']
        output = [o0, o1, o2, o3, o4, o5, o6, o7]

        p0 = csvd['pwm_limited[0]']
        p1 = csvd['pwm_limited[1]']
        p2 = csvd['pwm_limited[2]']
        p3 = csvd['pwm_limited[3]']
        p4 = csvd['pwm_limited[4]']
        p5 = csvd['pwm_limited[5]']
        p6 = csvd['pwm_limited[6]']
        p7 = csvd['pwm_limited[7]']
        pwm_limited = [p0, p1, p2, p3, p4, p5, p6, p7]

        return [csvname, x, status, controls, output, pwm_limited]

    @staticmethod
    def get_actuator_controls_0_0(ulgfile, tmpdir):
        csvname = 'actuator_controls_0_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['control[0]']
        y1 = csvd['control[1]']
        y2 = csvd['control[2]']
        y3 = csvd['control[3]']

        return [csvname, x, y0, y1, y2, y3]

    @staticmethod
    def get_manual_control_setpoint_0(ulgfile, tmpdir):
        csvname = 'manual_control_setpoint_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['x']
        y1 = csvd['y']
        y2 = csvd['z']
        y3 = csvd['r']

        return [csvname, x, y0, y1, y2, y3]

    @staticmethod
    def get_vehicle_local_position_0(ulgfile, tmpdir):
        csvname = 'vehicle_local_position_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['x']
        y1 = csvd['y']
        y2 = csvd['z']
        y3 = csvd['vx']
        y4 = csvd['vy']
        y5 = csvd['vz']
        y6 = csvd['ax']
        y7 = csvd['ay']
        y8 = csvd['az']

        return [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7, y8]

    @staticmethod
    def get_actuator_outputs_0(ulgfile, tmpdir):
        csvname = 'actuator_outputs_0'
        csvd = UlgParser.parse_csv(ulgfile, csvname, tmpdir)

        x = csvd['timestamp']
        y0 = csvd['output[0]']
        y1 = csvd['output[1]']
        y2 = csvd['output[2]']
        y3 = csvd['output[3]']
        y4 = csvd['output[4]']
        y5 = csvd['output[5]']
        y6 = csvd['output[6]']
        y7 = csvd['output[7]']

        return [csvname, x, y0, y1, y2, y3, y4, y5, y6, y7]

    @staticmethod
    def get_pandas_dataframe(tmpdir, ulgfile, csvname):
        ulgfile = FileFolderTools.get_file_basename(ulgfile)
        csvfile = ulgfile.replace('.ulg', '_') + csvname + '.csv'
        csvfile = tmpdir + '/' + csvfile
        df = pd.read_csv(csvfile, sep=',', index_col='timestamp')
        return df

    @staticmethod
    def get_pandas_dataframe_pos_vel(tmpdir, ulgfile, time_win):
        csvname = 'vehicle_local_position_0'
        df_pv = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_pv['vnorm'] = np.linalg.norm([df_pv['vx'].values, df_pv['vy'].values, df_pv['vz'].values], axis=0)
        df_pv['pnorm'] = np.linalg.norm([df_pv['x'].values, df_pv['y'].values, df_pv['z'].values], axis=0)
        df_pv = PandasTools.convert_index_from_us_to_s(df_pv)
        df_pv = PandasTools.apply_time_win(df_pv, time_win)
        return df_pv

    @staticmethod
    def get_pandas_dataframe_rpy_angles(tmpdir, ulgfile, time_win):
        csvname = 'vehicle_attitude_0_deg'
        df_att = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_att = PandasTools.convert_index_from_us_to_s(df_att)
        df_att = PandasTools.apply_time_win(df_att, time_win)

        csvname = 'vehicle_attitude_setpoint_0'
        df_attsp = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_attsp['roll sp'] = df_attsp['roll_body'].values * 180 / np.pi
        df_attsp['pitch sp'] = df_attsp['pitch_body'].values * 180 / np.pi
        df_attsp['yaw sp'] = df_attsp['yaw_body'].values * 180 / np.pi
        df_attsp = PandasTools.convert_index_from_us_to_s(df_attsp)
        df_attsp = PandasTools.apply_time_win(df_attsp, time_win)

        return [df_att, df_attsp]

    @staticmethod
    def get_pandas_dataframe_pqr_angvel(tmpdir, ulgfile, time_win):
        csvname = 'vehicle_angular_velocity_0'
        df_angvel = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_angvel['roll rate'] = df_angvel['xyz[0]'].values * 180 / np.pi
        df_angvel['pitch rate'] = df_angvel['xyz[1]'].values * 180 / np.pi
        df_angvel['yaw rate'] = df_angvel['xyz[2]'].values * 180 / np.pi
        df_angvel = PandasTools.convert_index_from_us_to_s(df_angvel)
        df_angvel = PandasTools.apply_time_win(df_angvel, time_win)

        csvname = 'vehicle_rates_setpoint_0'
        df_angvelsp = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_angvelsp['roll rate sp'] = df_angvelsp['roll'].values * 180 / np.pi
        df_angvelsp['pitch rate sp'] = df_angvelsp['pitch'].values * 180 / np.pi
        df_angvelsp['yaw rate sp'] = df_angvelsp['yaw'].values * 180 / np.pi
        df_angvelsp = PandasTools.convert_index_from_us_to_s(df_angvelsp)
        df_angvelsp = PandasTools.apply_time_win(df_angvelsp, time_win)

        return [df_angvel, df_angvelsp]

    @staticmethod
    def get_pandas_dataframe_man_ctrl(tmpdir, ulgfile, time_win):
        csvname = 'manual_control_setpoint_0'
        df_sticks = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_sticks.rename(columns={"x": "roll stick", "y": "pitch stick", "z": "throttle stick", 'r': "yaw stick"},
                         inplace=True)
        df_sticks = PandasTools.convert_index_from_us_to_s(df_sticks)
        df_sticks = PandasTools.apply_time_win(df_sticks, time_win)

        csvname = 'manual_control_switches_0'
        df_switches = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_switches = PandasTools.convert_index_from_us_to_s(df_switches)
        df_switches = PandasTools.apply_time_win(df_switches, time_win)

        return [df_sticks, df_switches]

    @staticmethod
    def get_pandas_dataframe_ctrl_alloc(tmpdir, ulgfile, time_win):
        csvname = 'actuator_controls_0_0'
        df_in = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_in.rename(columns={"control[0]": "roll rate cmd", "control[1]": "pitch rate cmd",
                              "control[2]": "yaw rate cmd", 'control[3]': "az cmd"},
                     inplace=True)
        df_in = PandasTools.convert_index_from_us_to_s(df_in)
        df_in = PandasTools.apply_time_win(df_in, time_win)

        # csvname = 'actuator_outputs_0'
        csvname = 'actuator_outputs_1'
        df_out = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        df_out = PandasTools.convert_index_from_us_to_s(df_out)
        df_out = PandasTools.apply_time_win(df_out, time_win)

        return [df_in, df_out]