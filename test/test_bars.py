#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from lasp.plot import BarScene
from PySide import QtGui
from PySide.QtCore import QTimer
from lasp.lasp_gui_tools import Branding, ASCEEColors
import numpy as np
import PySide.QtOpenGL as gl


def main():
    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    app.setFont(Branding.font())
    pix = QtGui.QPixmap(':img/img/lasp_logo_640.png')
    splash = QtGui.QSplashScreen(pixmap=pix)
    splash.show()
    mw = QtGui.QGraphicsView()
    glwidget = gl.QGLWidget()
    mw.setViewport(glwidget)
    bs = BarScene(None, np.array([10, 20, 300]), 2, ylim=(0, 1))
    mw.setScene(bs)

    bs.set_ydata(np.array([[.1, .2],
                           [.7, .8],
                           [.9, 1]]))

    # timer = QTimer.
    print(ASCEEColors.bggreen.getRgb())
    mw.show()                         # Show the form
    splash.finish(mw)
    app.exec_()                         # and execute the app


if __name__ == '__main__':
    main()                    # run the main function
