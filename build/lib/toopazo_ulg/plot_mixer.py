#!/usr/bin/env python

# from toopazo_tools.fileFolderTools import FileFolderTools
# from toopazo_tools.statistics import TimeseriesStats
from toopazo_tools.matplotlib import PlotTools, FigureTools

from toopazo_ulg.file_parser import UlgParser
from toopazo_ulg.plot_basics import UlgPlotBasics

import numpy as np
import scipy.linalg as scipy_linalg


class UlgPlotMixer(UlgPlotBasics):
    """
    The purpose of plot_mixer.py is to run a linear fit based on
    least square error for
        input = xvect = actuator_outputs
        output = yvect = actuator_controls

    The important function is
        [lsq_matrix, lsq_bias, lsq_error] =
        UlgPlotMixer.least_square_fit(xvect, yvect)

    And the prediction is output = lsq_matrix*input + lsq_bias

    The topic toopazo_ctrlalloc contains both the input and output and is used
    in the calculations

    firefly_cifer       = used by plot_sysid.py
    firefly_mixer       = used by plot_mixer.py for estimation/evaluation
    housefly_mixer1     = used by plot_mixer.py for estimation/evaluation
    housefly_mixer2     = used by plot_mixer.py for estimation/evaluation

    """

    @staticmethod
    def check_data(data, vmin, vmax):
        # print('[check_data] type(data) %s' % type(data))
        data = np.array(data)
        shape_tuple = data.shape
        if len(shape_tuple) == 1:
            UlgPlotMixer.check_data1d(data, vmin, vmax)
        else:
            UlgPlotMixer.check_data2d(data, vmin, vmax)

    @staticmethod
    def check_data2d(data2d, vmin, vmax):
        data2d = np.array(data2d)
        shape_tuple = data2d.shape
        ldim0 = shape_tuple[0]
        # ldim1 = shape_tuple[1]
        # print('data2d.shape %s' % str(data2d.shape))

        for ith_variable in range(0, ldim0):
            data1d = data2d[ith_variable]
            UlgPlotMixer.check_data1d(data1d, vmin, vmax)

    @staticmethod
    def check_data1d(data1d, vmin, vmax):
        for jth_sample in range(0, len(data1d)):
            data = data1d[jth_sample]
            if data > vmax:
                print('[check_data2d] sample %s: %s > vmax = %s' %
                      (jth_sample, data, vmax))
                # data1d[jth_sample] =
            if data < vmin:
                print('[check_data2d] sample %s: %s < vmin %s' %
                      (jth_sample, data, vmin))
                # data1d[jth_sample] =
            if data is np.nan:
                print('[check_data2d] sample %s: %s is np.nan' %
                      (jth_sample, data))
                # data1d[jth_sample] =
            # if jth_sample in range(1990, +2000):
            #     print('[check_data2d] ith_variable %s, sample %s: %s '
            #           % (ith_variable, jth_sample, data))
        # new_xvect.append(data1d)
    # return new_xvect

    @staticmethod
    def select_submatrix(xvect, s0, s1, v0, v1):
        new_xvect = []
        for ith_variable in range(v0, v1):
            data_arr = xvect[ith_variable]
            new_xvect.append(data_arr[s0:s1])
        return new_xvect
        # return np.array(new_xvect).T

    @staticmethod
    def get_stored_lsq_fit(vehicle, units):
        if (vehicle == 'firefly') and (units == 'output'):
            lsq_matrix = [
                [-1.4142, +1.4142, +2.0000, +2.0000],
                [+1.4142, +1.4142, -2.0000, +2.0000],
                [+1.4142, -1.4142, +2.0000, +2.0000],
                [-1.4142, -1.4142, -2.0000, +2.0000],
                [+1.4142, +1.4142, +2.0000, +2.0000],
                [-1.4142, +1.4142, -2.0000, +2.0000],
                [-1.4142, -1.4142, +2.0000, +2.0000],
                [+1.4142, -1.4142, -2.0000, +2.0000]
            ]
            lsq_bias = [-1, -1, -1, -1, -1, -1, -1, -1]
            return [lsq_matrix, lsq_bias]
        if (vehicle == 'firefly') and (units == 'pwm_limited'):
            lsq_matrix = [
                [-530, +530, +750, +750],
                [+530, +530, -750, +750],
                [+530, -530, +750, +750],
                [-530, -530, -750, +750],
                [+530, +530, +750, +750],
                [-530, +530, -750, +750],
                [-530, -530, +750, +750],
                [+530, -530, -750, +750]
            ]
            lsq_bias = [+1200, +1200, +1200, +1200, +1200, +1200, +1200, +1200]
            return [lsq_matrix, lsq_bias]
        if (vehicle == 'housefly') and (units == 'output'):
            print('inside housefly and output')
            lsq_matrix = [
                [-1.4142, +1.4142, +2.0000, +2.0000],
                [+1.4142, -1.4142, +2.0000, +2.0000],
                [+1.4142, +1.4142, -2.0000, +2.0000],
                [-1.4142, -1.4142, -2.0000, +2.0000]
            ]
            lsq_bias = [-1, -1, -1, -1]
            return [lsq_matrix, lsq_bias]
        if (vehicle == 'housefly') and (units == 'pwm_limited'):
            print('inside housefly and pwm_limited')
            lsq_matrix = [
                [-530, +530, +750, +750],
                [+530, -530, +750, +750],
                [+530, +530, -750, +750],
                [-530, -530, -750, +750]
            ]
            lsq_bias = [+1200, +1200, +1200, +1200]
            return [lsq_matrix, lsq_bias]
        # New mixers
        if (vehicle == 'firefly_mixer') and (units == 'output'):
            lsq_matrix = [
                [-1.4142, +1.4142, +2.0000, +2.0000],
                [+1.4142, +1.4142, -2.0000, +2.0000],
                [+1.4142, -1.4142, +2.0000, +2.0000],
                [-1.4142, -1.4142, -2.0000, +2.0000],
                [+1.4142, +1.4142, +2.0000, +2.0000],
                [-1.4142, +1.4142, -2.0000, +2.0000],
                [-1.4142, -1.4142, +2.0000, +2.0000],
                [+1.4142, -1.4142, -2.0000, +2.0000]
            ]
            lsq_bias = [-1, -1, -1, -1]
            return [lsq_matrix, lsq_bias]

    def mixer_input_output(self, ulgfile, closefig):
        [csvname, x, status, controls, output, pwm_limited] = \
            UlgParser.get_toopazo_ctrlalloc_0(ulgfile, self.tmpdir)
        _ = status

        config_lsq_fit = False
        # config_lsq_fit = True
        # config_vehicle = 'housefly'
        config_vehicle = 'firefly'
        config_output_units = 'output'
        # config_output_units = 'pwm_limited'

        if config_vehicle == 'firefly':
            noutputs = 8
        elif config_vehicle == 'housefly':
            noutputs = 4
        else:
            raise RuntimeError

        xvect = np.array(controls)
        if config_output_units == 'output':
            yvect = np.array(output)
        elif config_output_units == 'pwm_limited':
            yvect = np.array(pwm_limited)
        else:
            raise RuntimeError

        # Shortened log for least_square_fit calculation
        if config_lsq_fit:
            # sample0 = 2570  # houselfy log_148_2020-12-11-17-44-16.ulg
            # sample1 = 5000  # houselfy log_148_2020-12-11-17-44-16.ulg
            sample0 = 5000    # firefly log_134_2020-12-18-11-49-04.ulg
            sample1 = 8000    # firefly log_134_2020-12-18-11-49-04.ulg
        # Entire log
        else:
            sample0 = 0
            sample1 = len(xvect[0])

        tvect = x[sample0:sample1]

        variable0 = 0
        variable1 = 4  # [roll, pitch, yaw, thrust] => always 4 variables
        xvect = UlgPlotMixer.select_submatrix(
            xvect, sample0, sample1, variable0, variable1)

        variable0 = 0
        variable1 = noutputs
        yvect = UlgPlotMixer.select_submatrix(
            yvect, sample0, sample1, variable0, variable1)

        print('[mixer_input_output] check_data(xvect, -1, 1)')
        UlgPlotMixer.check_data(xvect, -1, 1)
        if config_output_units == 'output':
            print('[mixer_input_output] check_data(yvect, -2, 2)')
            UlgPlotMixer.check_data(yvect, -2, 2)
        elif config_output_units == 'pwm_limited':
            print('[mixer_input_output] check_data(yvect, 900, +2000)')
            UlgPlotMixer.check_data(yvect, 900, +2000)
        else:
            raise RuntimeError

        if config_lsq_fit:
            [lsq_matrix, lsq_bias, lsq_error] = \
                UlgPlotMixer.least_square_fit(xvect, yvect)
            _ = lsq_error
        else:
            [lsq_matrix, lsq_bias] = \
                UlgPlotMixer.get_stored_lsq_fit(
                    config_vehicle, config_output_units)

        [xvect, yvect, eval_yvect, eval_rms_error] = \
            UlgPlotMixer.least_square_eval(xvect, yvect, lsq_matrix, lsq_bias)
        _ = eval_rms_error

        print('[mixer_input_output] xvect.shape %s' % str(xvect.shape))
        print('[mixer_input_output] yvect.shape %s' % str(yvect.shape))
        print('[mixer_input_output] eval_yvect.shape %s'
              % str(eval_yvect.shape))
        print('[mixer_input_output] eval_yvect.shape %s'
              % str(eval_yvect.shape))

        for i in range(variable0, variable1):
            csvname_i = csvname + ('_%s' % i)

            # Plot lsq evauation
            [fig, ax_arr] = FigureTools.create_fig_axes(1, 1)
            fig.suptitle('Timeseries: actuator_controls_0_0')

            xlabel = 'timestamp s'
            x = tvect
            y_arr = [yvect[i, :], eval_yvect[i, :]]
            ylabel_arr = ['px4', 'lsq']
            PlotTools.ax1_x1_y2(ax_arr, x, xlabel, y_arr, ylabel_arr)
            if config_output_units == 'output':
                ax_arr[0].set_ylim([-2, +2])
                # pass
            if config_output_units == 'pwm_limited':
                ax_arr[0].set_ylim([700, +2000])
                # pass
            # ax1.tick_params(axis=u'y', which=u'both', length=0)

            # plt.show()
            jpgfilename = self.get_jpgfilename(
                self.plotdir, ulgfile, csvname_i)
            FigureTools.savefig(jpgfilename, closefig)

    @staticmethod
    def least_square_eval(xvect, yvect, lsq_matrix, lsq_bias):
        # arg = """
        # # UlgPlotMixer.least_square_eval(xvect, yvect)
        # #   xvect = [x0, x1, x2, x3]
        # #   yvect = [y0, y1, y2, y3, y4, y5, y6, y7]
        # #   It is assumed that xi and yj are arrays with nsamples
        # """
        # print(arg)

        nsamples = len(xvect[0])
        ninputs = len(xvect)
        noutputs = len(yvect)
        print('[least_square_eval] nsamples %s, ninputs %s, noutputs %s' %
              (nsamples, ninputs, noutputs))

        xvect = np.array(xvect)     # .reshape((nsamples, ninputs))
        yvect = np.array(yvect)     # .reshape((nsamples, noutputs))
        print('[least_square_eval] xvect.shape %s' % str(xvect.shape))
        print('[least_square_eval] yvect.shape %s' % str(yvect.shape))

        eval_yvect = np.zeros(yvect.shape)
        eval_sqerror = []
        # for ith_sample in range(0, +2):
        for ith_sample in range(0, nsamples):
            ith_input = xvect[:, ith_sample]
            ith_output = yvect[:, ith_sample]

            eval_output = np.matmul(lsq_matrix, ith_input) + lsq_bias
            eval_yvect[:, ith_sample] = eval_output

            sqerror = np.linalg.norm(eval_output - ith_output)
            eval_sqerror.append(sqerror)

        eval_rms_error = np.sqrt(np.mean(eval_sqerror))
        print('[least_square_eval] eval_rms_error %s' % eval_rms_error)

        return [xvect, yvect, eval_yvect, eval_rms_error]

    @staticmethod
    def least_square_fit(xvect, yvect):
        # arg = """
        # # UlgPlotMixer.least_square_fit(xvect, yvect)
        # #   xvect = [x0, x1, x2, x3]
        # #   yvect = [y0, y1, y2, y3, y4, y5, y6, y7]
        # #   It is assumed that xi and yj are arrays with nsamples
        # """
        # print(arg)

        nsamples = len(xvect[0])
        ninputs = len(xvect)
        noutputs = len(yvect)
        identity_noutputs = np.identity(noutputs)
        print('[least_square_fit] nsamples %s, ninputs %s, noutputs %s' %
              (nsamples, ninputs, noutputs))

        xvect = np.array(xvect)     # .reshape((nsamples, ninputs))
        yvect = np.array(yvect)     # .reshape((nsamples, noutputs))
        print('[least_square_fit] xvect.shape %s' % str(xvect.shape))
        print('[least_square_fit] yvect.shape %s' % str(yvect.shape))

        # Least Square Error solution to a multidimensional linear fit
        # yvect = lsq_matrix*xvect + lsq_bias + lsq_error
        # lsq_matrix = [a, b, c, d] = 4 vectors of 8x1
        # lsq_bias = vector of 8x1
        #
        # yvect[i] =
        # a*y0[i] + b*y1[i] +  c*y2[i] + d*y3[i] + lsq_bias + lsq_error[i]
        #
        # lsq_yvect = lsq_phi * lsq_xvect + lsq_error

        lsq_yvect = None
        lsq_phi = None
        # for ith_sample in range(0, +2):
        for ith_sample in range(0, nsamples):
            ith_input = xvect[:, ith_sample]
            ith_output = yvect[:, ith_sample]
            # print("ith_input %s" % ith_input)
            # print("ith_output %s" % ith_output)

            # 1) lsq_yvect
            if ith_sample == 0:
                lsq_yvect = np.array(ith_output)
            else:
                lsq_yvect = np.hstack((lsq_yvect, ith_output))

            # 2) Stack xvect part to lsq_phi
            ith_blockrow = None
            for jth_input_index in range(0, ninputs):
                jth_input_value = ith_input[jth_input_index]
                ith_jth_block = (jth_input_value * identity_noutputs)
                if jth_input_index == 0:
                    ith_blockrow = np.array(ith_jth_block)
                else:
                    ith_blockrow = np.hstack((ith_blockrow, ith_jth_block))
                # print(ith_blockrow.shape)

            # 3) Stack lsq_bias part to lsq_phi
            ith_blockrow = np.hstack((ith_blockrow, identity_noutputs))
            # print(ith_blockrow.shape)

            # lsq_phi.append(ith_blockrow)
            if ith_sample == 0:
                lsq_phi = ith_blockrow
            else:
                lsq_phi = np.vstack((lsq_phi, ith_blockrow))

            # print('lsq_yvect.shape %s' % str(lsq_yvect.shape))
            # print('lsq_phi.shape %s' % str(lsq_phi.shape))

        # Remove redundant dimensions (if any)
        lsq_phi = np.array(lsq_phi).squeeze()
        print('[least_square_fit] lsq_phi.shape %s' % str(lsq_phi.shape))

        # Check data validity
        print('[least_square_fit] check_data(lsq_phi, -10000, 10000)')
        UlgPlotMixer.check_data(lsq_phi, -10000, 10000)
        print('[least_square_fit] check_data(lsq_yvect, -10000, 10000)')
        UlgPlotMixer.check_data(lsq_yvect, -10000, 10000)

        # Find pseudo inverse matrix
        # lsq_pinvphi = np.linalg.pinv(lsq_phi)
        lsq_pinvphi = scipy_linalg.pinv(lsq_phi)
        lsq_pinvphi = np.array(lsq_pinvphi)
        print('[least_square_fit] lsq_pinvphi.shape %s'
              % str(lsq_pinvphi.shape))

        lsq_xvect = np.matmul(lsq_pinvphi, lsq_yvect)
        print('[least_square_fit] lsq_xvect.shape %s' % str(lsq_xvect.shape))
        # print('lsq_xvect %s' % str(lsq_xvect))

        lsq_error = np.linalg.norm(lsq_yvect - np.matmul(lsq_phi, lsq_xvect))
        print('[least_square_fit] lsq_error %s' % lsq_error)

        # Get lsq_matrix matrix back from lsq_xvect
        lsq_matrix = None
        for jth_input_index in range(0, ninputs):
            i0 = noutputs * jth_input_index
            i1 = i0 + noutputs
            jth_col_matrix = lsq_xvect[i0:i1]
            # print('jth_col_matrix %s' % jth_col_matrix)
            if jth_input_index == 0:
                lsq_matrix = np.array(jth_col_matrix)
            else:
                lsq_matrix = np.vstack((lsq_matrix, jth_col_matrix))
        lsq_matrix = lsq_matrix.T
        print('[least_square_fit] lsq_matrix.shape %s' % str(lsq_matrix.shape))
        print('[least_square_fit] lsq_matrix =\n%s' % lsq_matrix)
        lsq_bias = lsq_xvect[noutputs * ninputs:]
        print('[least_square_fit] lsq_bias =\n%s' % lsq_bias)

        return [lsq_matrix, lsq_bias, lsq_error]

    @staticmethod
    def print_2d_mtrix(matrix):
        # # s = [[str(e) for e in row] for row in matrix]
        # s = [[str(round(e, 6)) for e in row] for row in matrix]
        # lens = [max(map(len, col)) for col in zip(*s)]
        # # fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        # fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
        # table = [fmt.format(*row) for row in s]
        # print('\n'.join(table))

        print(matrix)

        # print(DataFrame(matrix))


if __name__ == '__main__':
    pass
