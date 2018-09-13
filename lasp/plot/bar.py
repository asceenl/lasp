#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.A. de Jong - ASCEE

Description:
Class for plotting bars on a QGraphicsScene.

"""
from ..lasp_gui_tools import ASCEEColors
from PySide.QtGui import (
    QGraphicsScene, QPen, QBrush, QGraphicsRectItem,
    QGraphicsTextItem, QPainter, QImage, QPrinter
    )

# from PySide.QtOpenGL import
from PySide.QtCore import Qt, QRectF, QLineF, QSize, QRect, QPointF, QSizeF
import numpy as np
import os


leftoffset = 120  # Left offset of the figure
rightoffset = 60
topoffset = 30
bottomoffset = 80
xticklabeloffset = 55
xlabelbottomoffset = 30
ylabelleftoffset = 30
nyticks = 11
ticklength = 10

# Distance between two bar groups in units of bar thicknesses
dxbars = 2

DEFAULT_COLORS = [ASCEEColors.blue, ASCEEColors.green, Qt.red]


class BarScene(QGraphicsScene):
    """
    Graphhics Scene for plotting bars
    """

    def __init__(self, parent, xvals, G, ylim=(0, 1),
                 grid=True,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 colors=DEFAULT_COLORS, size=(1200, 600),
                 legend=None,
                 legendpos=None):
        """
        Initialize a bar scene

        Args:
            xvals: labels and x positions of the bars
            G: Number of bars per x value
            ylim: y limits of the figure
            xlabel: label below x-axis
            ylabel: label on left side of the y-axis
            title: figure title
            colors: color cycler
            size: size of the plot in pixels
            legend: list of legend strings to show.
            legendpos: position of legend w.r.t. default position, in pixels
        """
        super().__init__(parent=parent)
        self.setSceneRect(QRect(0,0,*size))

        # self.setBackgroundBrush(ASCEEColors.bgBrush(0, size[0]))
        self.ylim = ylim
        N = len(xvals)
        self.N = N
        self.G = G
        self.bgs = []

        self.size = size
        xsize, ysize = size

        self.xsize = xsize
        self.ysize = ysize

        self.colors = colors

        # Size of the frame
        Lx = xsize - rightoffset - leftoffset
        Ly = ysize - topoffset - bottomoffset

        # The main frame where the bars are in.
        mainframe = self.createRect(leftoffset,
                                    bottomoffset,
                                    Lx,
                                    Ly)
        # Set the y ticks and ticklabels
        self.yticks = []
        txtmaxwidth = 0
        for i in range(nyticks):
            y = bottomoffset+Ly*i/(nyticks-1)

            ytick = self.addLine(leftoffset,
                                 y,
                                 leftoffset-ticklength,
                                 y)
            if grid:
                ygrid = self.addLine(leftoffset,
                                     y,
                                     xsize-rightoffset,
                                     y, pen=QPen(Qt.gray))

            range_ = ylim[1]-ylim[0]
            ytickval = i/(nyticks-1)*range_ + ylim[0]
            yticklabel = f'{ytickval:3.3}'
            txt = QGraphicsTextItem(yticklabel)
            txtwidth = txt.boundingRect().width()
            txtmaxwidth = max(txtmaxwidth, txtwidth)
            txt.setPos(leftoffset-10-txtwidth,
                       ysize - y-.022*self.ysize)
            self.addItem(txt)
            self.yticks.append(ytick)

        # Main frame added after grid lines, to get the color right
        self.addItem(mainframe)

        # # Create the bars
        for g in range(G):
            bg = []
            for n in range(N):
                barrect = self.getBarRect(n, g, 0)
                baritem = QGraphicsRectItem(barrect, brush=QBrush(Qt.blue))

                self.addItem(baritem)
                bg.append(baritem)

            self.bgs.append(bg)

        # Add x ticks and ticklabels
        xticklabels = []
        for n in range(N):
            xticklabel = f'{xvals[n]}'
            txt = QGraphicsTextItem(xticklabel)
            txtxpos = self.getBarGroupMidPos(n)-12
            txt.setPos(txtxpos,
                       self.ysize-bottomoffset+xticklabeloffset)
            txt.rotate(-90)
            self.addItem(txt)
            xticklabels.append(txt)

        # Set xlabel
        if xlabel is not None:
            xlabel = QGraphicsTextItem(xlabel)
            width = xlabel.boundingRect().width()
            txtxpos = xsize/2-width/2
            txtypos = ysize - xlabelbottomoffset
            xlabel.setPos(txtxpos, txtypos)
            self.addItem(xlabel)

        # # Set ylabel
        if ylabel is not None:
            ylabel = QGraphicsTextItem(ylabel)
            ylabel.setPos(ylabelleftoffset,
                          (ysize-topoffset-bottomoffset)/2+topoffset)
            ylabel.rotate(-90)
            self.addItem(ylabel)

        # Set title
        if title is not None:
            title = QGraphicsTextItem(title)
            width = xlabel.boundingRect().width()
            txtxpos = self.xsize/2-width/2
            txtypos = (1-.998)*self.ysize
            title.setPos(txtxpos, txtypos)
            self.addItem(title)

        if legend is not None:
            maxlegtxtwidth = 0
            legposx = 0 if legendpos is None else legendpos[0]
            legposy = 0 if legendpos is None else legendpos[1]

            legpos = (xsize-rightoffset-300+legposx,
                      ysize-topoffset-30+legposy)

            dyleg = 15
            dylegtxt = dyleg
            Lylegrect = 10
            Lxlegrect = 20
            legrectmargin = 5
            boxtopleft = QPointF(legpos[0]-legrectmargin,
                                 ysize-legpos[1]-Lylegrect-legrectmargin)

            legbox = self.addRect(QRectF(0, 0, 0, 0),
                                  pen=QPen(), brush=QBrush(Qt.white))

            for i, leg in enumerate(legend):
                leglabel = legend[i]

                # The position of the legend, in screen coordinates
                pos = (legpos[0], legpos[1] - i*dyleg)
                color = self.colors[i]

                legrect = self.createRect(*pos, Lxlegrect, Lylegrect)

                legrect.setBrush(QBrush(color))
                legtxt = QGraphicsTextItem(leglabel)
                maxlegtxtwidth = max(maxlegtxtwidth,
                                     legtxt.boundingRect().width())

                self.addItem(legrect)
                self.addItem(legtxt)

                legtxt.setPos(legpos[0]+Lxlegrect,
                              ysize-pos[1]-dylegtxt-3)

            legboxsize = QSize(maxlegtxtwidth+Lxlegrect+2*legrectmargin,
                               (i+1)*dyleg+legrectmargin)
            legboxrect = QRectF(boxtopleft, legboxsize)
            legbox.setRect(legboxrect)

    def saveAsBitmap(self, fn):
        """
        Save bar image as a jpg file. Overwrites a file already existing in
        filesystem.

        https://stackoverflow.com/questions/7451183/how-to-create-image-file\
        -from-qgraphicsscene-qgraphicsview#11642517

        Args:
            fn: Filename
        Returns:
            True on success
        """
        image = QImage(*self.size,
                       QImage.Format_ARGB32_Premultiplied)

        painter = QPainter(image)
        # painter.begin()
        # painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)
        painter.setPen(Qt.white)
        painter.drawRect(QRect(0, 0, *self.size))

        targetrect = QRectF(0, 0, *self.size)
        sourcerect = QRectF(0, 0, *self.size)
        self.render(painter, targetrect, sourcerect)
        painter.end()

        return image.save(fn)

    def saveAsPdf(self, fn, force=False):
        """
        Save bar image as a eps file.

        Args:
            fn: Filename
            force: if True, overwrites an existing file. If false, raises a
            RuntimeError if file already exists.
        """
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(fn)
        printer.setFullPage(True)
        printer.setPageSize(QPrinter.Custom)
        printer.setPaperSize(QSizeF(*self.size), QPrinter.Millimeter)
        printer.setPageMargins(0, 0, 0, 0, QPrinter.Millimeter)

        painter = QPainter(printer)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)
        painter.setPen(Qt.white)
        painter.drawRect(QRect(0, 0, *self.size))

        targetrect = QRectF(0, 0, printer.width(), printer.height())
        sourcerect = QRectF(0, 0, *self.size)
        self.render(painter, targetrect, sourcerect)
        painter.end()
        return True

    def getBarGroupMidPos(self, n):
        """
        Returns the mid x position below each bar group
        """
        Lx = self.xsize-rightoffset-leftoffset
        # Ly = self.ysize - topoffset - bottomoffset

        start = 10
        S = Lx - 2*start
        L = S/(self.N*self.G+dxbars*(self.N-1))
        xL = leftoffset+start
        return (n*(self.G*L+dxbars*L) + xL + self.G*L/2)

    def getBarRect(self, n, g, yval):
        """
        Returns a bar QRectF.

        Args:
            n: Bar index (i.e. corresponding to a certain frequency band)
            g: Bar group (i.e. corresponding to a certain quantity)
            yval: Height of bar, 1 for full lenght, 0 for no length

        Returns:
            QRectF corresponding to the bar at the right place in the scene
        """
        assert yval >= 0 and yval <= 1, "Invalid yval"
        Lx = self.xsize-rightoffset-leftoffset
        Ly = self.ysize-topoffset - bottomoffset

        start = 10
        S = Lx - 2*start
        assert S > 0, "Size of bar field is too small."
        # Width of a single bar
        L = S/(self.N*self.G+dxbars*(self.N-1))
        xL = leftoffset+start
        x = g*L + n*(self.G*L+dxbars*L) + xL

        return QRectF(x,
                      self.ysize-bottomoffset-yval*Ly,
                      L,
                      yval*Ly)

    def addLine(self, x1, y1, x2, y2, pen=QPen(), brush=QBrush()):
        line = QLineF(x1,
                      self.ysize - y1,
                      x2,
                      self.ysize - y2)
        return super().addLine(line, pen=pen, brush=brush)

    def createRect(self, x, y, Lx, Ly, pen=QPen(), brush=QBrush()):
        """
        Create a rectangle somewhere, in relative coordinates originating
        from the lower left position.
        """
        x1 = x

        # Y-position from the top, these are the coordinates used to create a
        # rect item.
        y1 = self.ysize-y-Ly
        return QGraphicsRectItem(x1,
                                 y1,
                                 Lx,
                                 Ly,
                                 pen=pen,
                                 brush=brush)

    def set_ydata(self, newydata):
        G = len(self.bgs)
        N = len(self.bgs[0])

        assert newydata.shape[0] == N
        assert newydata.shape[1] == G

        # Y-values of the bars should be between 0 and 1.
        scalefac = self.ylim[1]-self.ylim[0]
        yvals = (newydata - self.ylim[0])/scalefac

        # Clip values to be between 0 and 1
        yvals = np.clip(yvals, 0, 1)

        for g in range(G):
            color = self.colors[g]
            for n in range(N):
                bar = self.bgs[g][n]
                bar.setRect(self.getBarRect(n, g, yvals[n, g]))
                bar.setBrush(color)
