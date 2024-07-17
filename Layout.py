from PyQt6.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PyQt6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap,QRadialGradient,QPen,QActionGroup, QAction)
from PyQt6.QtWidgets import *
import sys
from pathlib import Path
import os
from CanvasLabel import MyCanvas



class UiLayout(object):
    def setupUiLayout(self, mainWindow):
        
        self.centralWidget = QWidget()

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")

        self.inputVerticalLayout = QVBoxLayout()
        self.inputVerticalLayout.setObjectName(u"inputVerticalLayout")
        self.textInput = QTextEdit(self)
        self.textInput.setFixedWidth(150)
        self.textInput.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        self.textInput.setObjectName(u"textInput")
        self.inputVerticalLayout.addWidget(self.textInput)
        self.inputVerticalLayout.addStretch(1)
        self.thumbnail = QPixmap()
        self.imageLabel = QLabel(self)
        self.imageLabel.setPixmap(self.thumbnail)
        self.importButton = QPushButton('import', self) #add import button
        self.inputVerticalLayout.addWidget(self.importButton)
        self.inputVerticalLayout.addWidget(self.imageLabel, 0, Qt.AlignmentFlag.AlignCenter)
        self.generateButton = QPushButton('generate', self) #add generate button
        self.inputVerticalLayout.addStretch(1)
        self.inputVerticalLayout.addWidget(self.generateButton)
        self.inputVerticalLayout.addStretch(5)
        self.horizontalLayout.addLayout(self.inputVerticalLayout)



        self.graphicsView = MyCanvas()
        self.graphicsView.setObjectName(u"graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)

        self.verticalLayout = QVBoxLayout()
        self.scrollAera = QScrollArea()
        self.scrollAera.setWidgetResizable(True)
        self.scrollAera.setFixedWidth(165)
        self.contentWidget = QWidget()
        self.gridLayout = QGridLayout(self.contentWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize) #reduce the spacing between the images

        self.buttonGroup = QButtonGroup()
        self.buttonGroup.setExclusive(True)
        self.dirPath = 'candidates'
        self.cddList = os.listdir(self.dirPath)
        for cdd in self.cddList:
            self.cddPath = os.path.join(self.dirPath, cdd)
            (self.cddPath)
            self.cddButton = QPushButton()
            self.cddButton.setCheckable(True)
            self.cddButton.setIconSize(QSize(150, 100))
            self.cddButton.setIcon(QIcon(self.cddPath))
            self.buttonGroup.addButton(self.cddButton)
            self.gridLayout.addWidget(self.cddButton)

        self.verticalLayout.addWidget(self.scrollAera)
        


        self.horizontalLayout.addLayout(self.verticalLayout)
        centralWidget = QWidget()
        centralWidget.setLayout(self.horizontalLayout)
        self.setCentralWidget(centralWidget)
        QMetaObject.connectSlotsByName(self) #UI工具生成代码，注释看好像也没影响
        

        #toolbar
        self.toolBarActoinGroup = QActionGroup(self)
        self.toolBarActoinGroup.setExclusive(True)

        self.penAct = QAction(QIcon('pen.png'), 'Pen', self)
        self.drawAct = QAction(QIcon('draw.png'), 'Draw', self)
        self.eraseAct = QAction(QIcon('eraser.png'), 'Erase', self)
        self.undoAct = QAction(QIcon('undo.png'), 'Undo', self)
        self.redoAct = QAction(QIcon('redo.png'), 'Redo', self)
        self.clearAct = QAction(QIcon('clear.png'), 'Clear', self)
        self.saveAct = QAction(QIcon('save.png'), 'Save', self)

        self.penAct.setCheckable(True)
        self.penAct.setChecked(False)
        self.drawAct.setCheckable(True)
        self.drawAct.setChecked(False)
        self.eraseAct.setCheckable(True)
        self.eraseAct.setChecked(False)
        self.toolbar = self.addToolBar('Draw')
        self.toolBarActoinGroup.addAction(self.penAct)
        self.toolBarActoinGroup.addAction(self.drawAct)
        self.toolBarActoinGroup.addAction(self.eraseAct)
        self.toolBarActoinGroup.addAction(self.undoAct)
        self.toolBarActoinGroup.addAction(self.redoAct)
        self.toolBarActoinGroup.addAction(self.clearAct)
        self.toolBarActoinGroup.addAction(self.saveAct)

        self.toolbar.addAction(self.penAct)
        self.toolbar.addAction(self.drawAct)
        self.toolbar.addAction(self.eraseAct)
        self.toolbar.addAction(self.undoAct)
        self.toolbar.addAction(self.redoAct)
        self.toolbar.addAction(self.clearAct)
        self.toolbar.addAction(self.saveAct)

            