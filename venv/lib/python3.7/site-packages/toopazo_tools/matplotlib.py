#!/usr/bin/env python

# from toopazo_tools.file_folder import FileFolderTools

import matplotlib.pyplot as plt
# import numpy as np


class FigureTools:
    def __init__(self):
        pass

    @staticmethod
    def create_fig_axes(a, b):
        # fig, (ax0, ax1, ax2) = plt.subplots(a, b, figsize=[7, 4])
        # ax_arr = [ax0, ax1, ax2]
        fig, ax_arr = plt.subplots(a, b, figsize=[8, 5])
        if (a == 1) and (b == 1):
            ax_arr = [ax_arr]
        return [fig, ax_arr]

    @staticmethod
    def savefig(filename, closefig):
        # plt.show()
        # plt.draw()

        # # Make fig the current figure
        # assert isinstance(fig, plt.figure())
        # plt.figure(fig.number)

        # Save and close current figure
        # fname = FileFolderTools.get_file_basename(filename)
        fname = filename
        print('[savefig] Saving %s' % fname)
        plt.savefig(filename, bbox_inches='tight')

        if closefig:
            plt.clf()
            plt.close()


class PlotTools:
    def __init__(self):
        pass

    # @staticmethod
    # def testfnct(*args, **kwargs):
    #     cnt = 0
    #     for item in args:
    #         print('cnt %s, item %s' % (cnt, item))
    #         cnt = cnt + 1
    #
    #     for key, value in kwargs.items():
    #         print('kwargs[%s] = %s' % (key, value))

    # @staticmethod
    # def add_vline(ax, x0, y0, ax_ylims, **kwargs):
    #     plt.sca(ax)
    #     plt.vlines(x=x0, ymin=ax_ylims[0], ymax= ax_ylims[1], **kwargs)
    #     # colors='k', linestyles='dashed', label='')
    #
    # @staticmethod
    # def add_text(x0, y0, arg, **kwargs):
    #     plt.text(x0, y0, arg, **kwargs)
    #     # fontsize = 12, color = 'black'

    @staticmethod
    def auto_lims(ax_arr, y_arr):
        # ymin = np.min(y_arr)
        # if ymin > 0:
        #     ylower = 0.8 * ymin
        # else:
        #     ylower = 1.2 * ymin
        #
        # ymax = np.max(y_arr)
        # if ymax > 0:
        #     yupper = 1.2 * ymax
        # else:
        #     yupper = 0.8 * ymax
        #
        # ax.set_ylim([ylower, yupper])
        _ = y_arr

        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y8(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[0], y_arr[1], color='green')
        ax_arr[0].plot(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].plot(x_arr[0], y_arr[3], color='black')
        ax_arr[0].plot(x_arr[0], y_arr[4], color='green', linestyle='dashed')
        ax_arr[0].plot(x_arr[0], y_arr[5], color='red', linestyle='dashed')
        ax_arr[0].plot(x_arr[0], y_arr[6], color='black', linestyle='dashed')
        ax_arr[0].plot(x_arr[0], y_arr[7], color='blue', linestyle='dashed')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # ax.ticklabel_format(useOffset=False)
        #   I was getting the error
        #   "This method only works with the ScalarFormatter.")
        #   AttributeError: This method only works with the ScalarFormatter.
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y1(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y2(ax, x, xlabel, y_arr, ylabel):
        ax = ax[0]  # for consistency with create_fig_axes
        # ax.hold(True)
        ax.grid(True)
        ax.plot(x, y_arr[0], color='red')
        ax.plot(x, y_arr[1], color='green')
        # ax.plot(x, y_arr[2], color='blue')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.ticklabel_format(useOffset=False)
        # UlgPlotFigures.auto_lims(ax, y_arr)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x2_y2(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[1], y_arr[1], color='green')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x2_y2_twinx(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        color = 'red'
        ax_arr[0].grid(True)
        ax_arr[0].set_xlabel(xlabel_arr[0])
        ax_arr[0].set_ylabel(ylabel_arr[0], color=color)
        ax_arr[0].plot(x_arr[0], y_arr[0], color=color)
        ax_arr[0].tick_params(axis='y', labelcolor=color)

        # instantiate a second axes that shares the same x-axis
        ax2 = ax_arr[0].twinx()

        # we already handled the x-label with ax1
        color = 'green'
        ax2.grid(True)
        ax2.set_ylabel(ylabel_arr[1], color=color)
        ax2.plot(x_arr[1], y_arr[1], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    @staticmethod
    def ax1_x3_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[1], y_arr[1], color='green')
        ax_arr[0].plot(x_arr[2], y_arr[2], color='blue')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x4_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[1], y_arr[1], color='green')
        ax_arr[0].plot(x_arr[2], y_arr[2], color='blue')
        ax_arr[0].plot(x_arr[3], y_arr[3], color='black')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[0], y_arr[1], color='green')
        ax_arr[0].plot(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].plot(x_arr[0], y_arr[3], color='black')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[0], y_arr[1], color='green')
        ax_arr[0].plot(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax2_x2_y2(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x_arr[1], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[1], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax3_x1_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x_arr[0], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x_arr[0], y_arr[2], color='red')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax3_x1_y6(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[0], y_arr[3], color='green')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x_arr[0], y_arr[1], color='red')
        ax_arr[1].plot(x_arr[0], y_arr[4], color='green')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x_arr[0], y_arr[2], color='red')
        ax_arr[2].plot(x_arr[0], y_arr[5], color='green')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax4_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x_arr[0], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x_arr[0], y_arr[2], color='red')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

        # ax_arr[3].hold(True)
        ax_arr[3].grid(True)
        ax_arr[3].plot(x_arr[0], y_arr[3], color='red')
        ax_arr[3].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[3])
        # PlotTools.auto_lims(ax_arr[3], y_arr[3])
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax4_x1_y8(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].plot(x_arr[0], y_arr[0], color='red')
        ax_arr[0].plot(x_arr[0], y_arr[4], color='green')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].plot(x_arr[0], y_arr[1], color='red')
        ax_arr[1].plot(x_arr[0], y_arr[5], color='green')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].plot(x_arr[0], y_arr[2], color='red')
        ax_arr[2].plot(x_arr[0], y_arr[6], color='green')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

        # ax_arr[3].hold(True)
        ax_arr[3].grid(True)
        ax_arr[3].plot(x_arr[0], y_arr[3], color='red')
        ax_arr[3].plot(x_arr[0], y_arr[7], color='green')
        ax_arr[3].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[3])
        # PlotTools.auto_lims(ax_arr[3], y_arr[3])
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)


class ScatterTools:
    def __init__(self):
        pass

    @staticmethod
    def auto_lims(ax_arr, y_arr):
        # ymin = np.min(y_arr)
        # if ymin > 0:
        #     ylower = 0.8 * ymin
        # else:
        #     ylower = 1.2 * ymin
        #
        # ymax = np.max(y_arr)
        # if ymax > 0:
        #     yupper = 1.2 * ymax
        # else:
        #     yupper = 0.8 * ymax
        #
        # ax.set_ylim([ylower, yupper])
        _ = y_arr

        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y8(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[0], y_arr[1], color='green')
        ax_arr[0].scatter(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].scatter(x_arr[0], y_arr[3], color='black')
        ax_arr[0].scatter(x_arr[0], y_arr[4], color='green', linestyle='dashed')
        ax_arr[0].scatter(x_arr[0], y_arr[5], color='red', linestyle='dashed')
        ax_arr[0].scatter(x_arr[0], y_arr[6], color='black', linestyle='dashed')
        ax_arr[0].scatter(x_arr[0], y_arr[7], color='blue', linestyle='dashed')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # ax.ticklabel_format(useOffset=False)
        #   I was getting the error
        #   "This method only works with the ScalarFormatter.")
        #   AttributeError: This method only works with the ScalarFormatter.
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y1(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x2_y2(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[1], y_arr[1], color='green')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x3_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[1], y_arr[1], color='green')
        ax_arr[0].scatter(x_arr[2], y_arr[2], color='blue')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x4_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[1], y_arr[1], color='green')
        ax_arr[0].scatter(x_arr[2], y_arr[2], color='blue')
        ax_arr[0].scatter(x_arr[3], y_arr[3], color='black')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[0], y_arr[1], color='green')
        ax_arr[0].scatter(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].scatter(x_arr[0], y_arr[3], color='black')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax1_x1_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax.hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[0], y_arr[1], color='green')
        ax_arr[0].scatter(x_arr[0], y_arr[2], color='blue')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        ax_arr[0].ticklabel_format(useOffset=False)
        # PlotTools.auto_lims(ax_arr, y_arr)
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax2_x2_y2(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].scatter(x_arr[1], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[1], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax3_x1_y3(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].scatter(x_arr[0], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].scatter(x_arr[0], y_arr[2], color='red')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax3_x1_y6(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].scatter(x_arr[0], y_arr[3], color='green')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].scatter(x_arr[0], y_arr[1], color='red')
        ax_arr[1].scatter(x_arr[0], y_arr[4], color='green')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].scatter(x_arr[0], y_arr[2], color='red')
        ax_arr[2].scatter(x_arr[0], y_arr[5], color='green')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

    @staticmethod
    def ax4_x1_y4(ax_arr, x_arr, xlabel_arr, y_arr, ylabel_arr):
        # ax_arr[0].hold(True)
        ax_arr[0].grid(True)
        ax_arr[0].scatter(x_arr[0], y_arr[0], color='red')
        ax_arr[0].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[0])
        # PlotTools.auto_lims(ax_arr[0], y_arr[0])
        ax_arr[0].locator_params(axis='y', nbins=3)
        ax_arr[0].locator_params(axis='x', nbins=4)

        # ax_arr[1].hold(True)
        ax_arr[1].grid(True)
        ax_arr[1].scatter(x_arr[0], y_arr[1], color='red')
        ax_arr[1].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[1])
        # PlotTools.auto_lims(ax_arr[1], y_arr[1])
        ax_arr[1].locator_params(axis='y', nbins=3)
        ax_arr[1].locator_params(axis='x', nbins=4)

        # ax_arr[2].hold(True)
        ax_arr[2].grid(True)
        ax_arr[2].scatter(x_arr[0], y_arr[2], color='red')
        ax_arr[2].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[2])
        # PlotTools.auto_lims(ax_arr[2], y_arr[2])
        ax_arr[2].locator_params(axis='y', nbins=3)
        ax_arr[2].locator_params(axis='x', nbins=4)

        # ax_arr[3].hold(True)
        ax_arr[3].grid(True)
        ax_arr[3].scatter(x_arr[0], y_arr[3], color='red')
        ax_arr[3].set(xlabel=xlabel_arr[0], ylabel=ylabel_arr[3])
        # PlotTools.auto_lims(ax_arr[3], y_arr[3])
        ax_arr[3].locator_params(axis='y', nbins=3)
        ax_arr[3].locator_params(axis='x', nbins=4)
