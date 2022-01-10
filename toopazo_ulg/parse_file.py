import subprocess
import csv
import numpy as np
import math
import pandas as pd
import os
import copy
import pandas
from toopazo_tools.file_folder import FileFolderTools
from toopazo_tools.pandas import PandasTools


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
    def check_ulog2csv(tmpdir, ulgfile):
        # Check if we need to run ulog2csv
        csvname = 'actuator_controls_0_0'
        csvfile = UlgParser.get_csvfile(tmpdir, ulgfile, csvname)
        # if FFTools.is_file(csvfile):
        if os.path.isfile(csvfile):
            # UlgParser.ulog2info(ulg_file)
            pass
        else:
            UlgParser.ulog2csv(ulgfile, tmpdir)
            UlgParser.write_vehicle_attitude_0_deg(ulgfile, tmpdir)

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
    def get_pos_vel_df(tmpdir, ulgfile):
        csvname = 'vehicle_local_position_0'
        pv_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        pv_df['vnorm'] = np.linalg.norm(
            [pv_df['vx'].values, pv_df['vy'].values, pv_df['vz'].values],
            axis=0
        )
        pv_df['pnorm'] = np.linalg.norm(
            [pv_df['x'].values, pv_df['y'].values, pv_df['z'].values],
            axis=0
        )
        pv_df = PandasTools.convert_index_from_us_to_s(pv_df)
        return pv_df

    @staticmethod
    def get_rpy_angles_df(tmpdir, ulgfile):
        csvname = 'vehicle_attitude_0_deg'
        att_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        att_df['angle norm'] = np.linalg.norm(
            [att_df['roll'].values,
             att_df['pitch'].values,
             att_df['yaw'].values],
            axis=0
        )
        att_df = PandasTools.convert_index_from_us_to_s(att_df)

        csvname = 'vehicle_attitude_setpoint_0'
        att_dfsp = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        att_dfsp['roll sp'] = att_dfsp['roll_body'].values * 180 / np.pi
        att_dfsp['pitch sp'] = att_dfsp['pitch_body'].values * 180 / np.pi
        att_dfsp['yaw sp'] = att_dfsp['yaw_body'].values * 180 / np.pi
        att_dfsp = PandasTools.convert_index_from_us_to_s(att_dfsp)

        return [att_df, att_dfsp]

    @staticmethod
    def get_pqr_angvel_df(tmpdir, ulgfile):
        csvname = 'vehicle_angular_velocity_0'
        angvel_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        angvel_df['roll rate'] = angvel_df['xyz[0]'].values * 180 / np.pi
        angvel_df['pitch rate'] = angvel_df['xyz[1]'].values * 180 / np.pi
        angvel_df['yaw rate'] = angvel_df['xyz[2]'].values * 180 / np.pi
        angvel_df['pqr norm'] = np.linalg.norm(
            [angvel_df['roll rate'].values,
             angvel_df['pitch rate'].values,
             angvel_df['yaw rate'].values],
            axis=0
        )
        angvel_df = PandasTools.convert_index_from_us_to_s(angvel_df)

        csvname = 'vehicle_rates_setpoint_0'
        angvelsp_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        angvelsp_df['roll rate sp'] = angvelsp_df['roll'].values * 180 / np.pi
        angvelsp_df['pitch rate sp'] = angvelsp_df['pitch'].values * 180 / np.pi
        angvelsp_df['yaw rate sp'] = angvelsp_df['yaw'].values * 180 / np.pi
        angvelsp_df = PandasTools.convert_index_from_us_to_s(angvelsp_df)

        return [angvel_df, angvelsp_df]

    @staticmethod
    def get_man_ctrl_df(tmpdir, ulgfile):
        csvname = 'manual_control_setpoint_0'
        sticks_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        sticks_df.rename(
            columns={
                "x": "roll stick",
                "y": "pitch stick",
                "z": "throttle stick",
                'r': "yaw stick"
            },
            inplace=True)
        sticks_df = PandasTools.convert_index_from_us_to_s(sticks_df)

        csvname = 'manual_control_switches_0'
        switches_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        switches_df = PandasTools.convert_index_from_us_to_s(switches_df)

        return [sticks_df, switches_df]

    @staticmethod
    def get_ctrl_alloc_df(tmpdir, ulgfile):
        csvname = 'actuator_controls_0_0'
        in_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        in_df.rename(
            columns={
                "control[0]": "roll rate cmd",
                "control[1]": "pitch rate cmd",
                "control[2]": "yaw rate cmd",
                'control[3]': "az cmd"
            },
            inplace=True)
        in_df = PandasTools.convert_index_from_us_to_s(in_df)

        # csvname = 'actuator_outputs_0'
        csvname = 'actuator_outputs_1'
        out_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        out_df = PandasTools.convert_index_from_us_to_s(out_df)

        return [in_df, out_df]

    @staticmethod
    def get_firefly_delta_df(tmpdir, ulgfile):
        csvname = 'firefly_delta'
        in_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        in_df.rename(columns={"control[0]": "roll rate cmd",
                              "control[1]": "pitch rate cmd",
                              "control[2]": "yaw rate cmd",
                              'control[3]': "az cmd"},
                     inplace=True)
        in_df = PandasTools.convert_index_from_us_to_s(in_df)

        # csvname = 'actuator_outputs_0'
        csvname = 'actuator_outputs_1'
        out_df = UlgParser.get_pandas_dataframe(tmpdir, ulgfile, csvname)
        out_df = PandasTools.convert_index_from_us_to_s(out_df)

        return [in_df, out_df]

    @staticmethod
    def get_ulg_dict(tmpdir, ulg_file):
        if not os.path.isfile(ulg_file):
            return None
        else:
            pv_df = UlgParser.get_pos_vel_df(
                tmpdir, ulg_file)
            [att_df, attsp_df] = UlgParser.get_rpy_angles_df(
                tmpdir, ulg_file)
            [angvel_df, angvelsp_df] = UlgParser.get_pqr_angvel_df(
                tmpdir, ulg_file)
            [sticks_df, switches_df] = UlgParser.get_man_ctrl_df(
                tmpdir, ulg_file)
            [in_df, out_df] = UlgParser.get_ctrl_alloc_df(tmpdir, ulg_file)

            ulg_df_dict = {
                'ulg_pv_df': pv_df,
                'ulg_att_df': att_df,
                'ulg_attsp_df': attsp_df,
                'ulg_angvel_df': angvel_df,
                'ulg_angvelsp_df': angvelsp_df,
                'ulg_sticks_df': sticks_df,
                'ulg_switches_df': switches_df,
                'ulg_in_df': in_df,
                'ulg_out_df': out_df,
            }
        return ulg_df_dict


class UlgParserTools:
    @staticmethod
    def synchronize_ulg_dict(ulg_dict, verbose):
        """
        :param ulg_dict: Dictionary of panda's dataframes to synchronize
        :param verbose: Print info for debug
        :return: Dataframes resampled at time_sec instants
        """
        assert isinstance(ulg_dict, dict)
        if verbose:
            print('Inside synchronize_ulg_dict')
            print('Before')
            for key, df in ulg_dict.items():
                # print(key)
                # print(df)
                mst = np.mean(np.diff(df.index))
                print('Mean sampling time of %s is %s s' % (key, mst))

        new_index = UlgParserTools.get_overlapping_index(ulg_dict, verbose)
        new_df_arr = UlgParserTools.resample_ulg_dict(ulg_dict, new_index)

        if verbose:
            print('After')
            for key, df in new_df_arr.items():
                # print(key)
                # print(df)
                mst = np.mean(np.diff(df.index))
                print('Mean sampling time of %s is %s s' % (key, mst))

        return copy.deepcopy(new_df_arr)

    @staticmethod
    def get_overlapping_index(ulg_dict, verbose):
        if verbose:
            print('Inside get_overlapping_index')
        t0_arr = []
        t1_arr = []
        ns_arr = []
        for key, df in ulg_dict.items():
            t0 = df.index[0]
            t1 = df.index[-1]
            ns = len(df.index)
            if verbose:
                print('df name %s, t0 %s, t1 %s, ns %s' % (key, t0, t1, ns))
            t0_arr.append(t0)
            t1_arr.append(t1)
            ns_arr.append(ns)

        t0_max = np.max(t0_arr)
        t1_min = np.min(t1_arr)
        ns_min = np.min(ns_arr)
        if t0_max < 0:
            raise RuntimeError
        if t1_min < t0_max:
            raise RuntimeError
        if ns_min <= 0:
            raise RuntimeError
        if verbose:
            print('t0_max %s, t1_min %s, ns_min %s' %
                  (t0_max, t1_min, ns_min))

        return np.linspace(t0_max, t1_min, ns_min)

    @staticmethod
    def resample_ulg_dict(ulg_dict, new_index):
        new_ulg_dict = {}
        x = new_index
        for key, ulg_df in ulg_dict.items():
            # xp = ulg_df.index
            # xp = DataframeTools.index_to_elapsed_time(df)
            # xp = df.index - df.index[0]
            xp = ulg_df.index
            data = UlgParserTools.get_data_dict_from_ulg_df(x, xp, ulg_df, key)
            new_df = pandas.DataFrame(data=data, index=x)
            new_ulg_dict[key] = new_df
            # print(f"key {key} ------------------------")
            # print(f"{ulg_df}")
            # print(f"{new_escid_df}")
        return copy.deepcopy(new_ulg_dict)

    # @staticmethod
    # def synchronize(ulg_dict, time_secs, verbose):
    #     """
    #     :param ulg_dict: Dictionary of panda's dataframes to synchronize
    #     :param time_secs: Array of time instants to resample (linear interop)
    #     :param verbose: Print info for debug
    #     :return: Dataframes resampled at time_sec instants
    #     """
    #     assert isinstance(ulg_dict, dict)
    #     if verbose:
    #         print('Synchronizing ulg_dict')
    #         for key, val in ulg_dict.items():
    #             print(key)
    #             print(val)
    #
    #     max_delta = 0.01
    #     if DataframeTools.check_time_difference(ulg_dict, max_delta):
    #         if verbose:
    #             print('DataframeTools.check_time_difference returned True')
    #             print('This means that the first and second sample times')
    #             print('are within %s' % max_delta)
    #             print('seconds, for all dataframes in ulg_dict')
    #         # new_index = DataframeTools.shortest_time_secs(ulg_dict)
    #         new_df_arr = UlgParserTools.resample(
    #             ulg_dict, time_secs, max_delta)
    #         return copy.deepcopy(new_df_arr)
    #     else:
    #         raise RuntimeError('UlgParserTools.check_time_difference failed')

    # @staticmethod
    # def resample(ulg_dict, time_secs, max_delta):
    #     if DataframeTools.check_time_difference(ulg_dict, max_delta):
    #         # new_index = DataframeTools.shortest_time_secs(ulg_dict)
    #         pass
    #     else:
    #         raise RuntimeError(
    #             'EscidParserTools.check_time_difference failed')
    #
    #     new_ulg_dict = {}
    #     x = time_secs
    #     for key, df in ulg_dict.items():
    #         # xp = ulg_df.index
    #         xp = DataframeTools.index_to_elapsed_time(df)
    #         data = UlgParserTools.get_data_dict_from_ulg_df(x, xp, df, key)
    #         index = x
    #         new_df = pandas.DataFrame(data=data, index=index)
    #         new_ulg_dict[key] = new_df
    #         # print(f"key {key} ------------------------")
    #         # print(f"{ulg_df}")
    #         # print(f"{new_escid_df}")
    #     return copy.deepcopy(new_ulg_dict)

    @staticmethod
    def get_data_dict_from_ulg_df(x, xp, ulg_df, ulg_type):
        if ulg_type == 'ulg_pv_df':
            data = {
                'x': np.interp(x, xp, fp=ulg_df['x']),
                'y': np.interp(x, xp, fp=ulg_df['y']),
                'z': np.interp(x, xp, fp=ulg_df['z']),
                'vx': np.interp(x, xp, fp=ulg_df['vx']),
                'vy': np.interp(x, xp, fp=ulg_df['vy']),
                'vz': np.interp(x, xp, fp=ulg_df['vz']),
                'vnorm': np.interp(x, xp, fp=ulg_df['vnorm']),
                'pnorm': np.interp(x, xp, fp=ulg_df['pnorm']),
            }
            return data
        if ulg_type == 'ulg_att_df':
            data = {
                'roll': np.interp(x, xp, fp=ulg_df['roll']),
                'pitch': np.interp(x, xp, fp=ulg_df['pitch']),
                'yaw': np.interp(x, xp, fp=ulg_df['yaw']),
            }
            return data
        if ulg_type == 'ulg_attsp_df':
            data = {
                'roll': np.interp(x, xp, fp=ulg_df['roll_body']),
                'pitch': np.interp(x, xp, fp=ulg_df['pitch_body']),
                'yaw': np.interp(x, xp, fp=ulg_df['yaw_body']),
                'angle norm': np.interp(x, xp, fp=ulg_df['angle norm']),
            }
            return data
        if ulg_type == 'ulg_angvel_df':
            data = {
                'roll rate': np.interp(x, xp, fp=ulg_df['roll rate']),
                'pitch rate': np.interp(x, xp, fp=ulg_df['pitch rate']),
                'yaw rate': np.interp(x, xp, fp=ulg_df['yaw rate']),
                'pqr norm': np.interp(x, xp, fp=ulg_df['pqr norm']),
            }
            return data
        if ulg_type == 'ulg_angvelsp_df':
            data = {
                'roll rate sp': np.interp(x, xp, fp=ulg_df['roll rate sp']),
                'pitch rate sp': np.interp(x, xp, fp=ulg_df['pitch rate sp']),
                'yaw rate sp': np.interp(x, xp, fp=ulg_df['yaw rate sp']),
            }
            return data
        if ulg_type == 'ulg_sticks_df':
            data = {
                'roll stick': np.interp(x, xp, fp=ulg_df['roll stick']),
                'pitch stick': np.interp(x, xp, fp=ulg_df['pitch stick']),
                'throttle stick': np.interp(x, xp, fp=ulg_df['throttle stick']),
                'yaw stick': np.interp(x, xp, fp=ulg_df['yaw stick']),
            }
            return data
        if ulg_type == 'ulg_switches_df':
            data = {
                'arm_switch': np.interp(x, xp, fp=ulg_df['arm_switch']),
                'return_switch': np.interp(x, xp, fp=ulg_df['return_switch']),
                'loiter_switch': np.interp(x, xp, fp=ulg_df['loiter_switch']),
                'offboard_switch': np.interp(x, xp, fp=ulg_df[
                    'offboard_switch']),
                'kill_switch': np.interp(x, xp, fp=ulg_df['kill_switch']),
                'gear_switch': np.interp(x, xp, fp=ulg_df['gear_switch']),
                'transition_switch': np.interp(x, xp, fp=ulg_df[
                    'transition_switch']),
                'mode_switch': np.interp(x, xp, fp=ulg_df['mode_switch']),
                'man_switch': np.interp(x, xp, fp=ulg_df['man_switch']),
                'acro_switch': np.interp(x, xp, fp=ulg_df['acro_switch']),
                'stab_switch': np.interp(x, xp, fp=ulg_df['stab_switch']),
                'posctl_switch': np.interp(x, xp, fp=ulg_df['posctl_switch']),
            }
            return data
        if ulg_type == 'ulg_in_df':
            data = {
                'roll rate cmd': np.interp(x, xp, fp=ulg_df['roll rate cmd']),
                'pitch rate cmd': np.interp(x, xp, fp=ulg_df['pitch rate cmd']),
                'yaw rate cmd': np.interp(x, xp, fp=ulg_df['yaw rate cmd']),
                'az cmd': np.interp(x, xp, fp=ulg_df['az cmd']),
            }
            return data
        if ulg_type == 'ulg_out_df':
            data = {
                'output[0]': np.interp(x, xp, fp=ulg_df['output[0]']),
                'output[1]': np.interp(x, xp, fp=ulg_df['output[1]']),
                'output[2]': np.interp(x, xp, fp=ulg_df['output[2]']),
                'output[3]': np.interp(x, xp, fp=ulg_df['output[3]']),
                'output[4]': np.interp(x, xp, fp=ulg_df['output[4]']),
                'output[5]': np.interp(x, xp, fp=ulg_df['output[5]']),
                'output[6]': np.interp(x, xp, fp=ulg_df['output[6]']),
                'output[7]': np.interp(x, xp, fp=ulg_df['output[7]']),
            }
            return data
        raise RuntimeError

    @staticmethod
    def remove_by_condition(ulg_dict, ulg_ref_cond):
        assert isinstance(ulg_dict, dict)
        new_ulg_dict = {}
        for key in ulg_dict.keys():
            ulg_df = ulg_dict[key]

            ulg_ref_cond.index = ulg_df.index
            ulg_df = ulg_df[ulg_ref_cond]

            new_ulg_dict[key] = ulg_df
        return copy.deepcopy(new_ulg_dict)

        # ulg_pv_df = ulg_dict['ulg_pv_df']
        # ulg_in_df = ulg_dict['ulg_in_df']
        # ulg_out_df = ulg_dict['ulg_out_df']
        #
        # # ulg_key = f'output[{int(reference_escid - 11)}]'
        # # ulg_ref_cond = ulg_out_df[ulg_key] > min_throttle
        #
        # ulg_ref_cond.index = ulg_pv_df.index
        # ulg_pv_df = ulg_pv_df[ulg_ref_cond]
        # ulg_ref_cond.index = ulg_in_df.index
        # ulg_in_df = ulg_in_df[ulg_ref_cond]
        # ulg_ref_cond.index = ulg_out_df.index
        # ulg_out_df = ulg_out_df[ulg_ref_cond]
        #
        # ulg_dict = {
        #     'ulg_pv_df': ulg_pv_df,
        #     # 'ulg_att_df': ulg_att_df,
        #     # 'ulg_attsp_df': ulg_attsp_df,
        #     # 'ulg_angvel_df': ulg_angvel_df,
        #     # 'ulg_angvelsp_df': ulg_angvelsp_df,
        #     # 'ulg_sticks_df': ulg_sticks_df,
        #     # 'ulg_switches_df': ulg_switches_df,
        #     'ulg_in_df': ulg_in_df,
        #     'ulg_out_df': ulg_out_df,
        # }
        # return copy.deepcopy(ulg_dict)
