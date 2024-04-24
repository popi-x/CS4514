import random
from math import factorial
from PyQt6 import QtWidgets, QtGui, QtCore

class ControlPoint(QtWidgets.QGraphicsObject):
    moved = QtCore.pyqtSignal(int, QtCore.QPointF)
    removeRequest = QtCore.pyqtSignal(object)

    brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

    # create a basic, simplified shape for the class
    _base = QtGui.QPainterPath()
    _base.addEllipse(-3, -3, 6, 6)
    _stroker = QtGui.QPainterPathStroker()
    _stroker.setWidth(30)
    _stroker.setDashPattern(QtCore.Qt.PenStyle.DashLine)
    _shape = _stroker.createStroke(_base).simplified()
    # "cache" the boundingRect for optimization
    _boundingRect = _shape.boundingRect()

    def __init__(self, index, pos, parent):
        super().__init__(parent)
        self.index = index
        self.setPos(pos)
        self.setFlags(
            self.GraphicsItemFlag.ItemIsSelectable 
            | self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemSendsGeometryChanges
            | self.GraphicsItemFlag.ItemStacksBehindParent
        )
        self.setZValue(-1)
        self.setToolTip(str(index + 1))
        self.font = QtGui.QFont()
        self.font.setBold(True)

    def setIndex(self, index):
        self.index = index
        self.setToolTip(str(index + 1))
        self.update()

    def shape(self):
        return self._shape

    def boundingRect(self):
        return self._boundingRect

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self.moved.emit(self.index, value)
        elif change == self.GraphicsItemChange.ItemSelectedHasChanged and value:
            # stack this item above other siblings when selected
            for other in self.parentItem().childItems():
                if isinstance(other, self.__class__):
                    other.stackBefore(self)
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu()
        removeAction = menu.addAction(QtGui.QIcon.fromTheme('edit-delete'), 'Delete point')
        if menu.exec(QtGui.QCursor.pos()) == removeAction:
            self.removeRequest.emit(self)

    def paint(self, qp, option, widget=None):
        qp.setBrush(self.brush)
        if not self.isSelected():
            qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawPath(self._shape)

        qp.setPen(QtCore.Qt.GlobalColor.white)
        qp.setFont(self.font)
        r = QtCore.QRectF(self.boundingRect())
        r.setSize(r.size() * 2 / 3)
        qp.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, str(self.index + 1))


class BezierItem(QtWidgets.QGraphicsPathItem):
    _precision = .05
    _delayUpdatePath = False
    _ctrlPrototype = ControlPoint
    def __init__(self, points=None):
        super().__init__()
        self.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.blue, 3, QtCore.Qt.PenStyle.DashLine))
        self.outlineItem = QtWidgets.QGraphicsPathItem(self)
        self.outlineItem.setFlag(self.GraphicsItemFlag.ItemStacksBehindParent)
        self.outlineItem.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 3, QtCore.Qt.PenStyle.DashLine))

        self.controlItems = []
        self._points = []

        if points is not None:
            self.setPoints(points)

    def setPoints(self, pointList):
        points = []
        for p in pointList:
            if isinstance(p, (QtCore.QPointF, QtCore.QPoint)):
                # always create a copy of each point!
                points.append(QtCore.QPointF(p))
            else:
                points.append(QtCore.QPointF(*p))
        if points == self._points:
            return

        self._points = []
        self.prepareGeometryChange()

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
        ctrlItem = self._ctrlPrototype(index, pos, self)
        self.controlItems.insert(index, ctrlItem)
        ctrlItem.moved.connect(self._controlPointMoved)
        ctrlItem.removeRequest.connect(self.removeControlPoint)

    def addControlPoint(self, pos):
        self.insertControlPoint(-1, pos)

    def insertControlPoint(self, index, pos):
        if index < 0:
            index = len(self._points)
        for other in self.controlItems[index:]:
            other.index += 1
            other.update()
        self._points.insert(index, pos)
        self._createControlPoint(index, pos)
        if not self._delayUpdatePath:
            self.updatePath()

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
        precision = max(.001, min(.5, precision))
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
            count = len(self._points)
            steps = round(count / self._precision)
            precision = 1 / steps
            n = count - 1
            # we're going to iterate through points *a lot* of times; with the
            # small cost of a tuple, we can cache the inner iterator to speed
            # things up a bit, instead of creating it in each for loop cycle
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
        self.setPath(self.curvePath)


class BezierExample(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.bezierScene = QtWidgets.QGraphicsScene()
        self.bezierView = QtWidgets.QGraphicsView(self.bezierScene)
        self.bezierView.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.bezierItem = BezierItem([
            (500, 500), (600, 700), (600, 550), (700, 500), 
            (700, 500), (800, 400), (1000, 200), (1000, 500)
        ])
        self.bezierScene.addItem(self.bezierItem)

        mainLayout = QtWidgets.QVBoxLayout(self)
        topLayout = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(topLayout)
        topLayout.addWidget(QtWidgets.QLabel('Resolution:'))

        resSpin = QtWidgets.QSpinBox(minimum=1, maximum=100)
        resSpin.setValue(self.bezierItem.stepRatio())
        topLayout.addWidget(resSpin)

        topLayout.addStretch()
        addButton = QtWidgets.QPushButton('Add point')
        topLayout.addWidget(addButton)

        mainLayout.addWidget(self.bezierView)

        self.bezierView.installEventFilter(self)
        resSpin.valueChanged.connect(self.bezierItem.setStepRatio)
        addButton.clicked.connect(self.addPoint)

    def addPoint(self, point=None):
        if not isinstance(point, (QtCore.QPoint, QtCore.QPointF)):
            point = QtCore.QPointF(
                random.randrange(int(self.bezierScene.sceneRect().width())), 
                random.randrange(int(self.bezierScene.sceneRect().height())))
        self.bezierItem.addControlPoint(point)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.MouseButtonDblClick:
            pos = self.bezierView.mapToScene(event.pos())
            self.addPoint(pos)
            return True
        return super().eventFilter(obj, event)

    def sizeHint(self):
        return QtWidgets.QApplication.primaryScreen().size() * 2 / 3


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = BezierExample()
    ex.show()
    app.exec()