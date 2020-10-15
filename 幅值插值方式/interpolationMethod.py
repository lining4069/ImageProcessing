import numpy as np

# 定义像素值取值函数
'''
输入:数字图像image,目标像素整型坐标(coordinate_x,coordinate_y)
输出:改坐标的像素值 gray_color
'''
def image_gray_color(image,coordinate_x,coordinate_y):
    return image[coordinate_y,coordinate_x]

# 定义最近邻插值方法
'''
输入:取像素值值图像image,(x,y)前向变换后获得的浮点坐标floatCoordinate
输出:该浮点坐标floatCoordinate的像素取值
'''
def nearest_neighbor_interpolation(image,floatCoordinate):
    u, v = floatCoordinate
    return image_gray_color(np.round(u),np.round(v))

# 定义双线性内插法 插值方式
'''
输入:取像素值值图像image,(x,y)前向变换后获得的浮点坐标floatCoordinate
输出:该浮点坐标floatCoordinate的像素取值
'''
def bilinear_interpolation(image,floatCoordinate):
    u, v = floatCoordinate
    i = int(u);
    j = int(v)
    a = u - i;
    b = v - j
    # 计算幅值
    # 双线性计算公式：f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)
    f=(1-a)*(1-b)*image_gray_color(i,j)+(1-a)*b*image_gray_color(i,j+1)+a*(1-b)*image_gray_color(i+1,j)+a*b*image_gray_color(i+1,j+1)
    # 返回np.uint8类型的返回值
    return np.array(f,dtype=np.uint8)

# 定义三次线性插值方式函数
'''
bincubic_interpolation函数：
输入:图像image,根据前向变换模型输出的浮点坐标 floatCoordinate=(u,v)
输入:结合配准后的图像的（x,y）处的像素值。
坐标对应关系:坐标(x,y)是前向变换模型的输入，也是几何配准中输出的像素的坐标
'''
# 定义双三次插值法中的权值函数
def W(d):
    d=abs(d)
    if(d<1):
        return 1-2*d**2+d**3
    if(d>=2):
        return 0
    return 4-8*d+5*d**2-d**3
# 实现bicubic_interpolation函数
def bicubic_interpolation(image,floatCoordinate):
    u,v=floatCoordinate
    # 确定x0,y0,xAlter,yAlter
    x0=int(u);y0=int(v)
    xAlter=u-x0;yAlter=v-y0
    # 估计Wy和Wx
    y=[1+yAlter,yAlter,1-yAlter,2-yAlter]
    x=[1+xAlter,xAlter,1-xAlter,2-xAlter]
    Wy=[];Wx=[]
    # 根据权值函数W()，为Wy和Wx中填充数据
    for yVetc in y:
        yres=W(yVetc)
        Wy.append(yres)
    for xVetc in x:
        xres=W(xVetc)
        Wx.append(xres)
    # 构建三次插值算法，参考的周围的16个参考点，像素取值矩阵F
    F=[
        [image_gray_color(x0-1,y0-1),image_gray_color(x0,y0-1),image_gray_color(x0+1,y0-1),image_gray_color(x0+2,y0-1)],
        [image_gray_color(x0-1,y0),image_gray_color(x0,y0),image_gray_color(x0+1,y0),image_gray_color(x0+2,y0)],
        [image_gray_color(x0-1,y0+1),image_gray_color(x0,y0+1),image_gray_color(x0+1,y0+1),image_gray_color(x0+2,y0+1)],
        [image_gray_color(x0-1,y0+2),image_gray_color(x0,y0+2),image_gray_color(x0+1,y0+2),image_gray_color(x0+2,y0+2)]
    ]
    # 根据输入图像以及权重向量Wy以及Wx。通过F来获取该变换后获得的浮点坐标的像素取值f
    f=np.mat(Wy)*np.mat(F)*(np.mat(Wx).T)
    # 返回np.unint8类型的像素值结果
    return np.array(f.A.flatten(),dtype=uint8)

# 定义调用不同灰度插值方式函数
'''
输入：
1、取像素值图像、
2、要求的像素的坐标（根据前行变换模型来计算参考图像（x,y）对应在输入图像的浮点坐标（u,v））
3、指定灰度插值方式，（最近邻插值-INTER_NEIGH，双线性插值方式-BILINEAR以及三次插值方式-CUBIC）
输出：该位置的应当获取的插值 ,返回的插值，数据类型为 np.uint8
'''
interpolation={0:"最近邻插值",1:"双线性插值",2:"三次插值"}

def get_back_grayscale_interpolation(image,floatCoordinate,interMethod):
    if(interMethod==0):
        return nearest_neighbor_interpolation(image,floatCoordinate)
    if(interMethod==1):
        return bilinear_interpolation(image,floatCoordinate)
    if(interMethod==2):
        return bicubic_interpolation(image,floatCoordinate)
    print("error for int interMethod")
    return 0