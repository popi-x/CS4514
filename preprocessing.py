import numpy as np
import matplotlib.pyplot as plt
import json
import os, sys
from sympy import *
from math import factorial
from scipy.special import comb
from scipy import spatial
from rdp import rdp

COEF = np.zeros((5, 5), dtype=np.float64)

#提取原json文件中的笔画id和path
def extractStrokes(fileName: str):
    with open(fileName) as f:
        data = json.load(f)
    strokes = []
    modified_strokes = cut_stroke(data)
    '''if errorList: #记录turning points错误的笔画id
        errorFile = open(errorListPath, 'a')
        errorFile.write('%s: %s\n' % (fileName, errorList))
        errorFile.close()'''

    '''with open('temp.json', 'w') as f:
        json.dump(modified_strokes, f)'''

    for stroke in modified_strokes['strokes']:
        if stroke['draw_type'] == 0:
            newStroke = {}
            newStroke['id'] = stroke['id']
            newStroke['path'] = stroke['path']
            strokes.append(newStroke)
        else:
            continue
    #print("Total strokes: ", totalStrokes)
    return strokes



def comb(n, r):
    if r == 0 or n == r:
        return 1
    else:
        return factorial(n) // (factorial(r) * factorial(n - r))


def generateBasisMatrix(n):
    M = np.zeros((n, n), dtype=np.float64)
    k = n-1
    for i in range(n):
        M[i][i] = comb(k, i)
    for i in range(n):
        for j in range(i+1, n):
            sign = 1 if (j-i) % 2 == 0 else -1
            value = comb(j, i) * M[j][j]
            M[j][i] = sign * value
    return M

def pdist_np(emb1, emb2): ##added by [deng]
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    # m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

## this function has been changed by [deng]
def angle(directions): ##added by [deng]
    """Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)

    ## added by [deng] arccos(-1) is invalid, -1 is 180 degree, so we set it to -0.9998
    for ele in range(len(cos)): 
        if cos[ele]<-0.998:
            cos[ele]=-0.9998

    return np.arccos(cos)



def compute_one_turning_point(trajectory): ##added by [deng]
    """ Compute the turning points of a given trajectory.
    """
    trajectory=np.array(trajectory)
    simplified_trajectory = rdp(trajectory, epsilon=2)
    sx, sy = simplified_trajectory.T

    # Compute the direction vectors on the simplified_trajectory.
    directions = np.diff(simplified_trajectory, axis=0)
    theta = angle(directions) # angle of all points in a samplified curve added by [deng]

    ## now we select top 3 points as the turning point as the turning point [added by deng]
    indices = np.argsort(theta, axis=0)[-3:] +1 # we rank the angle from small to large

    topk=[]
    for id in range(len(indices)):
        tp_x,tp_y =sx[indices[id]],sy[indices[id]]
        dist_mtx = pdist_np(trajectory, np.array([tp_x,tp_y])[np.newaxis, :])
        min_dist_idx = np.argmin(dist_mtx)
        topk.append(min_dist_idx)

    tp_final=topk[-1]
    middle_point_id=int(len(trajectory)/2)
    dist= np.abs(tp_final-middle_point_id)
    for tp_id in (topk):
        if np.abs(tp_id-int(len(trajectory)/2))<dist:## compute distance between turning point and the middle point of the stroke
            tp_final=tp_id
            dist=np.abs(tp_id-int(len(trajectory)/2))
        else:
            tp_final=tp_final

    return tp_final#min_dist_idx



## this function has been changed by [deng]
def compute_multiple_turning_point(trajectory): ##added by [deng]
    """ Compute the turning points of a given trajectory.
    """
    trajectory=np.array(trajectory)

    # Build simplified (approximated) trajectory
    # using RDP algorithm.
    simplified_trajectory = rdp(trajectory, epsilon=1)
    

    trajectory_mask = rdp(trajectory, epsilon=1, return_mask=True) #here is the mask of the simplified trajectory on original trajectory
    s2t=[ele for ele in range(len(trajectory)) if trajectory_mask[ele]==True] # here we get the index of the simplified trajectory on the original trajectory


    # Define a minimum angle to treat change in direction
    # as significant (valuable turning point).
    min_angle = np.pi / 5.0

    #-----------------------------------------------------------------------
    # Compute the direction vectors on the simplified_trajectory.
    directions = np.diff(simplified_trajectory, axis=0)
    theta = angle(directions) # angle of all points in a samplified curve added by [deng]

    # Select the index of the points with the greatest theta.
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta > min_angle)[0] + 1
    
    
    tp_final=np.array([s2t[ele] for ele in idx]) 

   
    return tp_final



def cut_stroke(data):
    """Cutting the long stroke into two parts based on the turning point.
    """
    modified_strokes = {}
    for s_ele in data:
        if s_ele!='strokes':
            modified_strokes[s_ele]=data[s_ele]
    
    strokes = data['strokes']
    modified_strokes['strokes']=[]

    for stroke in strokes:
        long_stroke=stroke['path']

        tp_ids = compute_multiple_turning_point(long_stroke)# compute multiple turning points
        strokes_ele={}
        for ele in stroke:
            if ele!= 'path' and 'pressure':
                strokes_ele[ele]=stroke[ele]
            
        tp_start=0

        '''if flag and stroke['draw_type'] == 0:
            errorList.append(stroke['id'])
            continue'''

        if len(tp_ids)==0:
            modified_strokes['strokes'].append(stroke)
            continue
        else:
            for tp_id in tp_ids:
                new_stroke=strokes_ele.copy()
                new_stroke['path']=long_stroke[tp_start:tp_id]
                tp_start=tp_id-1
                modified_strokes['strokes'].append(new_stroke)
                #if stroke['id'] == 112: print(new_stroke)

            new_stroke=strokes_ele.copy()
            new_stroke['path']=long_stroke[tp_id-1:]
            
            '''if stroke['id'] == 112:
                print('new_stroke:',new_stroke, 'tp_ids:',tp_ids)'''
            
            modified_strokes['strokes'].append(new_stroke)

      
    return modified_strokes



#计算控制点
def getControlPoints(points, degree):
    M = COEF
    #print(points)
    sameColList = np.all(points == points[0,:], axis=0)

    #如果所有点的x坐标或y坐标相同，则直接取线段的端点作为控制点
    if sameColList[0]:
        C = [[points[0,0], i] for i in np.linspace(points[0,1], points[-1,1], degree)]
        return np.array(C, dtype=np.float64)
    if sameColList[1]:
        C = [[i, points[0,1]] for i in np.linspace(points[0,0], points[-1,0], degree)]
        return np.array(C, dtype=np.float64)
    
    begin = points[0] #根据算法算出来的起点和笔画起点不一致（没有研究具体原因），直接取笔画起点，下同
    end = points[-1]
    P = points
    length = len(P)
    D = [0]
    for i in range(1, length):
        D.append(D[i-1] + np.linalg.norm(P[i] - P[i-1]))   
    S = [D[i] / D[-1] for i in range(length)]
    S = np.array(S)
    '''if np.isnan(S).any():
        print(points)'''
    T = np.tile(S, (degree, 1)).transpose()
    T = np.power(T, np.arange(degree))
    #因为有的矩阵没有逆矩阵，所以用伪逆矩阵（pinv）
    C = np.matmul(np.matmul(np.matmul(np.linalg.pinv(M), np.linalg.pinv(np.matmul(T.transpose(), T))), T.transpose()), P)
    C[0] = begin
    C[-1] = end
    #print(C)
    return C


#生成路径
def generatePath(points, degree, T):
    M = COEF
    C = getControlPoints(points, degree)
    #print(T.shape, M.shape, C.shape)
    path = np.matmul(np.matmul(T, M), C)
    return path

'''
def getRange(ran, p):
    pmax = np.max(p, axis=0)
    pmin = np.min(p, axis=0)
    #print(pmax, pmin)
    if ran[0][0] > pmin[0]:
        ran[0][0] = pmin[0]
    if ran[0][1] > pmin[1]:
        ran[0][1] = pmin[1]
    if ran[1][0] < pmax[0]:
        ran[1][0] = pmax[0]
    if ran[1][1] < pmax[1]:
        ran[1][1] = pmax[1]
    return ran
'''

#去除重复点
def removeDuplicate(points):
    newPoints = []
    for i in range(len(points)):
        if i == 0:
            newPoints.append(points[i])
        else:
            if np.linalg.norm(points[i] - points[i-1]) > 0:
                newPoints.append(points[i])
    return np.array(newPoints, dtype=np.float64)


'''def addPoints(points):
    insertInfo = {}
    length = len(points)
    diff = 15-length
    dis = [np.linalg.norm(points[i] - points[i-1]) for i in range(1, length)]
    totalDis = np.sum(dis)
    portion = dis / totalDis
    portion = list(enumerate(portion.tolist()))
    portion.sort(key=lambda x: x[1], reverse=True)
    while diff > 0:
        #print(diff)
        pointsNum = ceiling(ceiling(portion[0][1]) * diff)
        index = portion[0][0]
        insertPoints = np.linspace(points[index-1], points[index], pointsNum+2)
        insertInfo[index-1] = insertPoints[1:-1]
        diff -= pointsNum
        portion.pop(0)
    sortedInsertInfo = {key:insertInfo[key] for key in sorted(insertInfo.keys())} 
    flag = 0
    elements = list()
    for i in sortedInsertInfo:
        elements.extend(points[flag:i].tolist())
        elements.extend(sortedInsertInfo[i].tolist())
        flag = i+1  
    newPoints = np.stack(elements, axis=0)  

    #plt.plot(newPoints[:,0], newPoints[:,1], 'bo', markersize=3)

    return newPoints'''


#绘制笔画并记录控制点为json文件
def drawStrokes(t, fileDir, saveDir):
    strokes = extractStrokes(fileDir)

    #img = plt.imread('DifferSketching_Dataset/Industrial_Component/original_png/N004_2_0_MD_carter100K.png')
    plt.style.use('classic')
    plt.figure(figsize=(8, 8))

    #reduceP = [1,2,4,8]
    reduceP = [1]
    ran = [[100, 100], [0, 0]]
    for r in reduceP:
        rr = 800 / r
        
        CPrecord = {'control_points':[]} #control points record
        for degree in [5]:
            global COEF
            COEF = generateBasisMatrix(degree)
            alphaValue = 1
            color = 'k'

            for stroke in strokes:
                #print(stroke)
                points = stroke['path']
                #print(points)
                strokeCP = []
                if r == 1:   
                    points = np.array(points)
                else:
                    points = np.rint(np.array(points) / r) 
                if np.unique(points, axis=0).shape[0] == 1:
                    plt.plot(points[:1,0], points[:1,1], '%so' % color, markersize=0.1, alpha=alphaValue)
                    pointList = np.tile(points[0], (degree, 1)).tolist()
                    strokeCP.append(pointList)
                    CPrecord['control_points'].append(strokeCP)
                    continue

                T = np.tile(t, (degree, 1)).transpose()
                T = np.power(T, np.arange(degree))
                points = removeDuplicate(points)

                #plt.plot(points[:,0], points[:,1], 'ro')

                '''if True not in np.all(points == points[0,:], axis=0) and len(points) < degree: 
                    points = addPoints(points)'''
                bound = 200

                #超过一定数量的点，分段绘制
                if len(points) > bound:
                    cutNum = len(points) // bound
                    cutBound = len(points) // cutNum + 2
                    for i in range(0, len(points), cutBound-1):
                        p = points[i:i+cutBound]
                        if np.unique(p, axis=0).shape[0] == 1:
                            plt.plot(points[:1,0], points[:1,1], '%so' % color, markersize=0.1, alpha=alphaValue)
                            pointList = np.tile(points[0], (degree, 1)).tolist()
                            strokeCP.append(pointList)
                            continue
                        #ran = getRange(ran, p)
                        C = getControlPoints(p, degree)
                        strokeCP.append(C.tolist())
                        path = generatePath(p, degree, T)
                        px, py = path[:,0], path[:,1]
                        plt.plot(px, py, '%s-' % color)

                    CPrecord['control_points'].append(strokeCP)

                else:
                    #ran = getRange(ran, points)

                    C = getControlPoints(points, degree)
                    strokeCP.append(C.tolist())
                    CPrecord['control_points'].append(strokeCP)
                    
                    '''plt.plot(C[:,0], C[:,1], 'o')
                    for i in range(len(C)):
                        plt.annotate('%.2f, %.2f' % (C[i][0], C[i][1]), xy=(C[i][0], C[i][1]), fontsize = 7)'''
                    
                    path = generatePath(points, degree, T)
                    px, py = path[:,0], path[:,1]
                    #plt.annotate('(%.2f, %.2f)' % (px[0], py[0]), xy=(px[0], py[0]), xytext=(px[0]+0.1, py[0]+0.1), fontsize=5, color='black')
                    plt.plot(px, py, '%s-' % color, alpha = alphaValue)
                

        #print(CPrecord)
        fileName = os.path.splitext(os.path.basename(fileDir))[0]

        with open('%s/%djson/%s.json' % (saveDir, rr, fileName), 'w') as f:
            json.dump(CPrecord, f)

        #single test
        '''with open('%s/%s.json' % (saveDir, fileName), 'w') as f:
            json.dump(CPrecord, f)'''

        #plt.imshow(img, zorder = 0, extent=[0, 800, 800, 0])
        plt.xlim(0, 800)
        plt.ylim(0, 800)
        plt.gca().invert_yaxis()
        plt.axes().set_aspect('equal')
        plt.axis('off')
        plt.savefig('%s/%dimg/%s.png' % (saveDir, rr, fileName), facecolor='white')

        #single test
        #plt.savefig('%s/%s.png' % (saveDir, fileName), facecolor='white') 

        plt.clf()

    plt.close()
    #print('X: ', ran[0][0],'-',ran[1][0], '\nY: ', ran[0][1],'-',ran[1][1])


#批量处理
def batchExec():
    try:
        categoreis = ['Animal', 'Animal_Head', 'Chair', 'Human_Face', 'Industrial_Component', 'Lamp', 'Primitive', 'Shoe','Vehicle']
        for category in categoreis:
            fileDir = 'DifferSketching_Dataset/%s/original_json' % category
            fileList = os.listdir(fileDir)
            saveDir = 'dsNew/%s' % category
            os.makedirs(saveDir, exist_ok=True)
            os.makedirs('%s/%djson' % (saveDir, 800), exist_ok=True)
            os.makedirs('%s/%dimg' % (saveDir, 800), exist_ok=True)
            for fileName in fileList:
                filePath = fileDir + '/' + fileName
                #rang = [0, 0]

                drawStrokes(t, filePath, saveDir)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_obj, exc_tb.tb_lineno)
        print(filePath)

#注释部分为单个文件测试
if '__main__' == __name__:
    '''filePath = 'DifferSketching_Dataset/Primitive/original_json/N001_0_2_MD_ccylinder.json'
    saveDir =  'temp' '''
    step = 0.001 #设置时间步长
    t = np.arange(0,1,step)
    #drawStrokes(t, filePath, saveDir)
    batchExec()
