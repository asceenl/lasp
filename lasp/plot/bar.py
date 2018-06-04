#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.A. de Jong - ASCEE

Description:
Class for plotting bars on a QGraphicsScene.

"""
from ..lasp_gui_tools import ASCEEColors
from PySide.QtGui import (
    QGraphicsScene, QGraphicsView, QPen, QBrush, QGraphicsRectItem,
    QGraphicsTextItem, QPainter, QImage
    )

# from PySide.QtOpenGL import
from PySide.QtCore import Qt, QRectF, QLineF, QSize, QRect, QPointF
import numpy as np
import os


leftoffset = .1  # Left offset of the figure
rightoffset = 0
topoffset = .05
bottomoffset = .1
nyticks = 6
ticklength = .01

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
                 colors=DEFAULT_COLORS, size=(800, 600),
                 legend=None):
        """
        Initialize a bar scene

        Args:
            xvals: labels and x positions of the bars
            G: Number of bars per x value
            ylim: y limits of the figure


        """
        super().__init__(parent=parent)

        # self.setBackgroundBrush(ASCEEColors.bgBrush(0, size[0]))
        self.ylim = ylim
        N = len(xvals)
        self.N = N
        self.G = G
        self.bgs = []

        self.size = size
        self.colors = colors

        # Size of the frame
        Lx = 1 - rightoffset - leftoffset
        Ly = 1 - topoffset - bottomoffset

        # The main frame where the bars are in.
        mainframe = self.createRect(leftoffset,
                                    bottomoffset,
                                    Lx,
                                    Ly)
        self.addItem(mainframe)

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
                                     1-rightoffset,
                                     y,pen=QPen(Qt.gray))

            range_ = ylim[1]-ylim[0]
            ytickval = i/(nyticks-1)*range_ + ylim[0]
            yticklabel = f'{ytickval:2}'
            txt = QGraphicsTextItem(yticklabel)
            txtwidth = txt.boundingRect().width()
            txtmaxwidth = max(txtmaxwidth, txtwidth)
            txt.setPos((leftoffset-.03)*self.xscale-txtwidth,
                        (1-y-.022)*self.yscale)
            self.addItem(txt)
            self.yticks.append(ytick)

        # Create the bars
        for g in range(G):
            bg = []
            for n in range(N):
                barrect = self.getBarRect(n, g, 0)
                baritem = QGraphicsRectItem(barrect, brush=QBrush(Qt.blue))

                self.addItem(baritem)
                bg.append(baritem)

            self.bgs.append(bg)

        # Add x ticks and ticklabels
        for n in range(N):
            xticklabel = f'{xvals[n]}'
            txt = QGraphicsTextItem(xticklabel)
            txtxpos = self.getBarGroupMidPos(n)-0.01*self.xscale
            txt.setPos(txtxpos,
                       self.yscale*(1-bottomoffset+.1))
            txt.rotate(-90)
            self.addItem(txt)

        # Set xlabel
        if xlabel is not None:
            xlabel = QGraphicsTextItem(xlabel)
            width = xlabel.boundingRect().width()
            txtxpos = self.xscale/2-width/2
            txtypos = .998*self.yscale
            xlabel.setPos(txtxpos, txtypos)
            self.addItem(xlabel)

        # Set ylabel
        if ylabel is not None:
            ylabel = QGraphicsTextItem(ylabel)
            ylabel.setPos((leftoffset-.01)*self.xscale-txtmaxwidth,
                          ((1-topoffset-bottomoffset)/2+topoffset)*self.yscale)
            ylabel.rotate(-90)
            self.addItem(ylabel)


        # Set title
        if title is not None:
            title = QGraphicsTextItem(title)
            width = xlabel.boundingRect().width()
            txtxpos = self.xscale/2-width/2
            txtypos = (1-.998)*self.yscale
            title.setPos(txtxpos, txtypos)
            self.addItem(title)

        legpos = (1-rightoffset-.3, 1-topoffset-.05)

        dyleg = 0.03
        dylegtxt = dyleg
        Lyleg = .02
        Lxleg = .05
        legrectmarginpix = 5
        boxtopleft = QPointF(legpos[0]*self.xscale-legrectmarginpix,
                             (1-legpos[1]-Lyleg)*self.yscale-legrectmarginpix)


        if legend is not None:
            nlegs = len(legend)
            maxlegtxtwidth = 0
            for i,leg in enumerate(legend):
                leglabel = legend[i]

                # The position of the legend, in our coordinates
                pos = (legpos[0], legpos[1] - i*dyleg)
                color = self.colors[i]

                legrect = self.createRect(*pos,Lxleg,Lyleg)

                legrectwidth = legrect.boundingRect().width()

                legrect.setBrush(QBrush(color))
                legtxt = QGraphicsTextItem(leglabel)
                maxlegtxtwidth = max(maxlegtxtwidth,
                                     legtxt.boundingRect().width())

                self.addItem(legrect)
                self.addItem(legtxt)

                legtxt.setPos(legpos[0]*self.xscale+legrectwidth,
                              (1-pos[1]-dylegtxt)*self.yscale)

                boxbottomright = legtxt.boundingRect().topRight()

            legboxsize = QSize(maxlegtxtwidth+legrectwidth+2*legrectmarginpix,
                               (i+1)*dyleg*self.yscale+legrectmarginpix)

            legboxrect = QRectF(boxtopleft,legboxsize)
            legbox = self.addRect(legboxrect)


    def saveAsPng(self, fn, force=False):
        """
        Save bar image as a jpg file.

        https://stackoverflow.com/questions/7451183/how-to-create-image-file\
        -from-qgraphicsscene-qgraphicsview#11642517

        """
        if os.path.exists(fn) and not force:
            raise RuntimeError(f"File {fn} already exists in filesystem.")

        # self.clearSelection()
        image = QImage(*self.size,
                       QImage.Format_ARGB32_Premultiplied)
        # image = QImage()
        painter = QPainter(image)
        # painter.begin()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.white)
        painter.setPen(Qt.white)
        painter.drawRect(QRect(0,0,*self.size))

        targetrect = QRectF(0,0,*self.size)
        sourcerect = QRectF(0,0,*self.size)
        self.render(painter,targetrect,sourcerect)
        painter.end()
        # print('saving image')
        image.save(fn)

    def getBarGroupMidPos(self,n):
        """
        Returns the mid x position below each bar group
        """
        Lx = 1-rightoffset-leftoffset
        Ly = 1 - topoffset - bottomoffset

        start = .05
        S = Lx - 2*start
        L = S/(self.N*self.G+dxbars*(self.N-1))
        xL = leftoffset+start
        return (n*(self.G*L+dxbars*L) + xL + self.G*L/2)*self.xscale

    def getBarRect(self, n, g, yval):
        Lx = 1-rightoffset-leftoffset
        Ly = 1 - topoffset - bottomoffset

        start = .05
        S = Lx - 2*start
        L = S/(self.N*self.G+dxbars*(self.N-1))
        xL = leftoffset+start
        x = g*L + n*(self.G*L+dxbars*L) + xL

        return QRectF(x*self.xscale,
                      (1-bottomoffset-yval*Ly)*self.yscale,
                      L*self.xscale,
                      yval*Ly*self.yscale)

    def addLine(self, x1, y1, x2, y2, pen=QPen(), brush=QBrush()):
        line = QLineF(x1*self.xscale,
                      (1-y1)*self.yscale,
                      (x2)*self.xscale,
                      (1-y2)*self.yscale)
        return super().addLine(line, pen=pen, brush=brush)

    def createRect(self, x, y, Lx, Ly, pen=QPen(), brush=QBrush()):
        """
        Create a rectangle somewhere, in relative coordinates originating
        from the lower left position.
        """
        x1 = x

        # Y-position from the top, these are the coordinates used to create a
        # rect item.
        y1 = 1-y-Ly
        return QGraphicsRectItem(x1*self.xscale,
                                 y1*self.yscale,
                                 Lx*self.xscale,
                                 Ly*self.yscale,
                                 pen=pen,
                                 brush=brush)

    @property
    def xscale(self):
        return self.size[0]

    @property
    def yscale(self):
        return self.size[1]

    def set_ydata(self, newydata):
        G = len(self.bgs)
        N = len(self.bgs[0])

        assert newydata.shape[0] == N
        assert newydata.shape[1] == G

        # Crop values to be between 0 and 1
        scalefac = self.ylim[1]-self.ylim[0]
        yvals = np.clip(newydata, self.ylim[0], self.ylim[1])/scalefac

        for g in range(G):
            color = self.colors[g]
            for n in range(N):
                bar = self.bgs[g][n]
                bar.setRect(self.getBarRect(n, g, yvals[n, g]))
                bar.setBrush(color)
