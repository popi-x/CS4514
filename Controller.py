from PyQt6.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap,QRadialGradient,QPen,QPainterPath, QAction)
from PyQt6.QtWidgets import *
import sys
from pathlib import Path
from Layout import UiLayout
from CanvasLabel import BezierItem


class Controller(UiLayout, QMainWindow):

    def __init__(self):

        super(Controller, self).__init__()
        self.setupUiLayout(self)
        self.setWindowTitle("533")
        self.setGeometry(0, 0, 1200, 800)

        self.importButton.clicked.connect(self.importImage)
        self.penAct.toggled.connect(self.pen)
        self.drawAct.toggled.connect(self.draw)
        self.eraseAct.toggled.connect(self.erase)
        self.buttonGroup.buttonClicked.connect(self.imageClicked)
        self.generateButton.clicked.connect(self.showCandidates)
        
    def importImage(self): 
        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)

        if fname[0]:
            self.thumbnail.load(fname[0])
            self.thumbnail = self.thumbnail.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(self.thumbnail)

    def pen(self, checked):
        if checked:
            self.graphicsView.setPenType(1)
        else:
            self.graphicsView.setPenType(0)
            self.graphicsView.deselectAllCurves()

    def draw(self, checked):
        if checked:
            self.graphicsView.bezierScene.drawType = 1
        else:
            self.graphicsView.bezierScene.drawType = 0

    def erase(self, checked):
        if checked:
            self.graphicsView.bezierScene.drawType = 2
        else:
            self.graphicsView.bezierScene.drawType = 0
    
    
    def imageClicked(self, button):
        for btn in self.buttonGroup.buttons():
            btn.setStyleSheet('')

        self.graphicsView.showSVG()
        button.setStyleSheet("border: 2px solid blue;")

    def showCandidates(self):
        self.scrollAera.setWidget(self.contentWidget)


if __name__ == "__main__":
    app=QApplication(sys.argv)
    MyWin = Controller()
    MyWin.show()
    sys.exit(app.exec())