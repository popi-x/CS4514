import sys,random,time,json,copy,os
from math import factorial
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtGui, QtCore
from PyQt6.QtGui import (QBrush, QColor,QPen,QPainterPath, QAction, QUndoCommand, QUndoStack, QImage, QPainter)
from PyQt6.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
import numpy as np


class ControlPoint(QtWidgets.QGraphicsObject):
    moved = QtCore.pyqtSignal(int, QtCore.QPointF) #moved will receive an integer and a QPointF instance 
    removeRequest = QtCore.pyqtSignal(object)

    brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

    # create a basic, simplified shape for the class
    _base = QtGui.QPainterPath()
    _base.addEllipse(-7, -7, 13, 13)                   #this is the shape of control points
    _stroker = QtGui.QPainterPathStroker()             #generate fillable outlines for a given painter path
    _stroker.setWidth(30)                              #the size of control points?
    _stroker.setDashPattern(QtCore.Qt.PenStyle.DashLine)
    _shape = _stroker.createStroke(_base).simplified() #Returns a simplified version of this path, merging all subpaths that intersect and returning a path containing no intersecting edges
    # "cache" the boundingRect for optimization
    _boundingRect = _shape.boundingRect()              #Returns the bounding rectangle of this painter path as a rectangle with floating point precision.

    def __init__(self, index, pos, parent):
        super().__init__(parent)
        self.index = index
        self.setPos(pos)
        # All flags in flags are enabled; all flags not in flags are disabled.
        self.setFlags(
            self.GraphicsItemFlag.ItemIsSelectable 
            | self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemSendsGeometryChanges
            | self.GraphicsItemFlag.ItemStacksBehindParent
        )                 
        self.setZValue(-1) #the Z value is set to -1 to ensure that the control points are always drawn behind the curve
        self.font = QtGui.QFont()
        self.font.setBold(True)

    def setIndex(self, index):
        self.index = index
        self.update()

    def shape(self):
        return self._shape

    def boundingRect(self):
        return self._boundingRect


    def itemChange(self, change, value): #the type of value depends on the change
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self.moved.emit(self.index, value)
        elif change == self.GraphicsItemChange.ItemSelectedHasChanged and value:
            # stack this item above other siblings when selected 选中的点在其他点上面
            for other in self.parentItem().childItems():
                if isinstance(other, self.__class__):
                    other.stackBefore(self)
        return super().itemChange(change, value)


    #reimplement paint function
    def paint(self, qp, option, widget=None):
        qp.setBrush(self.brush) #brush is red
        if not self.isSelected():
            qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawPath(self._base) #draw new control points

class communicate(QtCore.QObject):
    editing = QtCore.pyqtSignal(int)

class BezierChange(QtCore.QObject):
    bezierChange = QtCore.pyqtSignal(int)

class BezierItem(QtWidgets.QGraphicsPathItem):
    _precision = .001
    _delayUpdatePath = False
    _ctrlPrototype = ControlPoint

    def __init__(self, index, points=None): #if points aren't passed, it will be None
        super().__init__()
        self.c = communicate()
        self.b = BezierChange()
        self.editing = self.c.editing
        self.bezierChange = self.b.bezierChange
        self.index = index
        self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, QtCore.Qt.PenStyle.SolidLine))
        self.outlineItem = QtWidgets.QGraphicsPathItem(self)
        self.outlineItem.setFlag(self.GraphicsItemFlag.ItemStacksBehindParent) #the child item is drawn behind the parent item (by default, child items are drawn on top of their parent item)
        self.outlineItem.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1, QtCore.Qt.PenStyle.DashLine))
        #self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.pathCoords = []
        self.mousePressEvent = self.selectedCurve
        self.editable = 0

        self.controlItems = []
        self._points = []

        if points is not None: #if points are passed in
            self.setPoints(points)
        

    def selectedCurve(self, event):
        if self.editable == 1:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                x = event.scenePos().x()
                y = event.scenePos().y()
                diff = np.array(self.pathCoords) - np.array([x, y])
                if np.linalg.norm(diff, axis=1).min() < 2:
                    self.editing.emit(self.index)
                    self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.green, 2, QtCore.Qt.PenStyle.SolidLine))
                    for ctrlItem in self.controlItems:
                        ctrlItem.setVisible(True)
                    self.outlineItem.setVisible(True)
            
    
    def deselect(self):
        self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, QtCore.Qt.PenStyle.SolidLine))
        for ctrlItem in self.controlItems:
            ctrlItem.setVisible(False)
        self.outlineItem.setVisible(False)
                

    def setPoints(self, pointList):
        points = []
        for p in pointList:
            if isinstance(p, (QtCore.QPointF, QtCore.QPoint)):
                # always create a copy of each point!
                points.append(QtCore.QPointF(p))
            else:
                points.append(QtCore.QPointF(*p)) #*p allows you to pass a list or tuple of arguments to a function
        if points == self._points: #？why
            return

        self._points = []
        self.prepareGeometryChange() #notifies the scene that the item's geometry is about to change


        #?clear controlItems to contain new control points?
        while self.controlItems:
            item = self.controlItems.pop()
            item.setParentItem(None)
            if self.scene():
                self.scene().removeItem(item)
            del item

        self._delayUpdatePath = True
        for i, p in enumerate(points):
            self.insertControlPoint(i, p)
        self._delayUpdatePath = False

        self.updatePath()

    def _createControlPoint(self, index, pos):
        self.ctrlItem = self._ctrlPrototype(index, pos, self) #_ctrlPrototype is ControlPoint class, the 'self'(BeizerItem) will be the parent of the controlPoint instance
        self.controlItems.insert(index, self.ctrlItem)
        self.ctrlItem.moved.connect(self._controlPointMoved)
        self.ctrlItem.removeRequest.connect(self.removeControlPoint)

        self.ctrlItem.setVisible(False)

    def addControlPoint(self, pos):
        self.insertControlPoint(-1, pos)

    def insertControlPoint(self, index, pos):
        if index < 0:
            index = len(self._points)
        for other in self.controlItems[index:]: #skip this loop when the first several points are inserted
            other.index += 1
            other.update()
        self._points.insert(index, pos) #insert the point at the specified index
        self._createControlPoint(index, pos)
        if not self._delayUpdatePath: #check if the last point is inserted
            self.updatePath()
        self.outlineItem.setVisible(False)



    def removeControlPoint(self, cp):
        if isinstance(cp, int):
            index = cp
        else:
            index = self.controlItems.index(cp)

        item = self.controlItems.pop(index)
        self.scene().removeItem(item)
        item.setParentItem(None)
        for other in self.controlItems[index:]:
            other.index -= 1
            other.update()

        del item, self._points[index]

        self.updatePath()

    def precision(self):
        return self._precision

    def setPrecision(self, precision):
        #precision = max(.001, min(.5, precision))
        precision = 0.001
        if self._precision != precision:
            self._precision = precision
            self._rebuildPath()

    def stepRatio(self):
        return int(1 / self._precision)

    def setStepRatio(self, ratio):
        '''
        Set the *approximate* number of steps per control point. Note that 
        the step count is adjusted to an integer ratio based on the number 
        of control points.
        '''
        self.setPrecision(1 / ratio)
        self.update()

    def updatePath(self):
        outlinePath = QtGui.QPainterPath()
        if self.controlItems:
            outlinePath.moveTo(self._points[0])
            for point in self._points[1:]:
                outlinePath.lineTo(point)
        self.outlineItem.setPath(outlinePath)
        self._rebuildPath()

    def _controlPointMoved(self, index, pos):
        self._points[index] = pos
        self.updatePath()
        self.currentBezierItem = selfCopy(self)
        self.bezierChange.emit(self.index)

    def _rebuildPath(self):
        '''
        Actually rebuild the path based on the control points and the selected
        curve precision. The default (0.05, ~20 steps per control point) is
        usually enough, lower values result in higher resolution but slower
        performance, and viceversa.
        '''
        self.curvePath = QtGui.QPainterPath()
        if self._points:
            self.curvePath.moveTo(self._points[0])
            self.pathCoords.append([self._points[0].x(), self._points[0].y()])
            count = len(self._points)
            steps = round(count / self._precision)
            precision = 1 / steps
            n = count - 1
            pointIterator = tuple(enumerate(self._points))
            for s in range(steps + 1):
                u = precision * s
                x = y = 0
                for i, point in pointIterator:
                    binu = (factorial(n) / (factorial(i) * factorial(n - i)) 
                        * (u ** i) * ((1 - u) ** (n - i)))
                    x += binu * point.x()
                    y += binu * point.y()
                self.curvePath.lineTo(x, y)
                self.pathCoords.append([x, y])
        self.setPath(self.curvePath)

def selfCopy(self):
    return BezierItem(self.index, self._points)




class MyCanvas(QtWidgets.QGraphicsView):

    def __init__(self):
        super().__init__()
       

        with open('sample/controlPoints.json', 'r') as f:
            data = json.load(f)
        
        self.cp = [j for i in data['control_points'] for j in i]

        self.bezierScene = CustomScene()
        self.setScene(self.bezierScene)
        self.setSceneRect(0, 0, 800, 800)

        '''index = 0
        for group in self.cp:
            self.controlPoints = group
            self.bezierItem = BezierItem(index, self.controlPoints)
            self.bezierItem.editing.connect(self.deselectCurves)
            self.bezierScene.addItem(self.bezierItem)
            index += 1'''

        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
       

    def deselectCurves(self, index):
        for item in self.bezierScene.items():
            if isinstance(item, BezierItem) and item.index != index:
                item.deselect()
    
    def deselectAllCurves(self):
        for item in self.bezierScene.items():
            if isinstance(item, BezierItem):
                item.deselect()

    def setPenType(self, penType):
        self.bezierScene.penType = penType
        for item in self.bezierScene.items():
            if isinstance(item, BezierItem):
                item.editable = penType


    
    def sizeHint(self):
        return QtWidgets.QApplication.primaryScreen().size() * 2 / 3
    
    def showSVG(self):
        index = 0

        for group in self.cp:
            self.controlPoints = group
            self.bezierItem = BezierItem(index, self.controlPoints)
            #self.bezierScene.currentBezierItems.append(self.bezierItem)
            self.bezierItem.editing.connect(self.deselectCurves)
            self.bezierScene.addItem(self.bezierItem)
            index += 1
        #self.BezierScene.lastBezierItems = copy.copy(self.BezierScene.currentBezierItems)

        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.bezierScene.currentItems = self.bezierScene.items()


class CustomScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.drawType = 0
        self.penType = 0
        self.drawPen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 2, QtCore.Qt.PenStyle.SolidLine)
        self.eraserPen = QtGui.QPen(QtCore.Qt.GlobalColor.gray, 3, QtCore.Qt.PenStyle.SolidLine)
        self.undoStack = QUndoStack()
        self.currentStrokeItems = []
        self.lastStrokeItems = []

    def mousePressEvent(self, event):
        if self.drawType == 0:
            super().mousePressEvent(event)
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.drawType == 2:
                self.eraserCoords = []
                self.eraserCoords.append([event.scenePos().x(), event.scenePos().y()])
                self.QEraserPath = QtWidgets.QGraphicsPathItem()  
                self.eraserPath = QtGui.QPainterPath()
                self.eraserPath.moveTo(event.scenePos())
                self.QEraserPath.setPen(self.eraserPen)
                self.addItem(self.QEraserPath)
            if self.drawType == 1:
                self.QGraphicsPath = QtWidgets.QGraphicsPathItem() 
                self.path1 = QtGui.QPainterPath()
                self.path1.moveTo(event.scenePos()) 
                self.QGraphicsPath.setPen(self.drawPen) 
                self.addItem(self.QGraphicsPath) 


    def mouseMoveEvent(self, event):
        if self.drawType == 0:
            super().mouseMoveEvent(event)
            return
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton: 
            if self.drawType == 2:
                self.eraserCoords.append([event.scenePos().x(), event.scenePos().y()])
                self.eraserPath.lineTo(event.scenePos())
                self.QEraserPath.setPath(self.eraserPath)
            if self.drawType == 1:
                if self.path1:
                    self.path1.lineTo(event.scenePos()) 
                    self.QGraphicsPath.setPath(self.path1) 

    def mouseReleaseEvent(self, event):
        if self.drawType == 0:
            super().mouseReleaseEvent(event)
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.drawType == 2:
                self.eraseItem()
                self.removeItem(self.QEraserPath)
                self.eraserPath = QPainterPath()
            if self.drawType == 1:
                if self.path1:
                    self.path1.closeSubpath() 
                    self.lastStrokeItems = copy.copy(self.currentStrokeItems)
                    self.currentStrokeItems.append(self.QGraphicsPath)
            addCom = AddCommand(self)
            self.undoStack.push(addCom)

    def isIntersected(self,item):
        for i in self.eraserCoords:
            diff = np.array(item.pathCoords) - np.array(i)
            if np.linalg.norm(diff, axis=1).min() < 2:
                return True
        return False

    def eraseItem(self):
        for item in self.items(self.eraserPath, QtCore.Qt.ItemSelectionMode.IntersectsItemShape):
            if item != self.QEraserPath:
                if not isinstance(item, BezierItem):
                    self.lastStrokeItems = copy.copy(self.currentStrokeItems)
                    self.currentStrokeItems.remove(item)
                    self.removeItem(item)
                else:
                    if self.isIntersected(item):
                        self.removeItem(item)
        return
        
            
    def undo(self):
        self.undoStack.undo()

    def redo(self):
        self.undoStack.redo()

    def saveScene(self):
        image = QImage(self.sceneRect().size().toSize(), QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.render(painter)
        painter.end()

        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)")
        if fileName:
            image.save(fileName)
    

                

class AddCommand(QUndoCommand):
    def __init__(self, scene):
        super(AddCommand, self).__init__()

        self.scene = scene

        self.lastStrokeItems = copy.copy(self.scene.lastStrokeItems)
        self.currentStrokeItems = copy.copy(self.scene.currentStrokeItems)
        

    def undo(self):
        for item in self.scene.currentStrokeItems:
            self.scene.removeItem(item)


        self.scene.currentStrokeItems = copy.copy(self.lastStrokeItems)
        self.scene.lastStrokeItems = copy.copy(self.lastStrokeItems)
        
        for item in self.scene.lastStrokeItems:
                self.scene.addItem(item)

    def redo(self):
        for item in self.scene.lastStrokeItems:
            self.scene.removeItem(item)

        self.scene.currentStrokeItems = copy.copy(self.currentStrokeItems)
        self.scene.lastStrokeItems = copy.copy(self.currentStrokeItems)
        
        for item in self.scene.currentStrokeItems:
                self.scene.addItem(item)

            
