from math import *
import cv2 as cv
import numpy as np

def xyzTo_uv(w, f, t, hc):
	"""根据标定信息(f,t,hc)，现实世界中点映射至图像中
	** w(np类型)代表世界坐标x,y,z, p(np类型)代表图像坐标u,v"""
	x, y, z = w
	a = y * cos(t) - z * sin(t) + hc * sin(t)
	p = [(f*x) / a, (f*hc*cos(t) - f*y*sin(t) - z*f*cos(t))/a]
	return np.array(p)

def uvTo_xyz(w, p, f, t, hc):
	"""根据标定信息(f,t,hc)及给定z，图像中点映至世界坐标系中, 注：w, p(np类型)"""
	z, u, v = w[2], p[0], p[1]
	w[1] = (f*hc*cos(t) - f*z*cos(t) + z*v*sin(t) - hc*v*sin(t)) / (v*cos(t) + f*sin(t))
	a = w[1] * cos(t) - z * sin(t) + hc * sin(t)
	w[0] = (a * u) / f
	return w

def rd_xyzTo_uv(w, p, f, t, pan, hc):
	"""根据标定信息(f,t,pan,hc)，世界点映射至图像点(注：世界坐标系平行于路面)"""
	x, y, z = w[0], w[1], w[2]
	a = sin(pan) * cos(t) * x + cos(t) * cos(pan) * y - sin(t) * z + hc * sin(t)
	p[0] = (f * cos(pan) * x - f * sin(pan) * y) / a
	p[1] = (-f*sin(pan) * sin(t)*x - f*sin(t)*cos(pan)*y - f*cos(t)*z + f*hc*cos(t)) / a

def rd_uvTo_xyz(w, p, f, t, pan, hc):
	"""根据标定信息(f,t,pan,hc)及给定z，图像中点映至世界坐标系中(注：世界坐标系平行于路面)"""
	z, u, v = w[2], p[0], p[1]
	k0 = (u * sin(pan) * cos(t) - f * cos(pan)) / (-f * sin(pan) - u * cos(t) * cos(pan))
	b0 = u * (-sin(t) * z + hc * sin(t)) / (-f * sin(pan) - u * cos(t) * cos(pan))
	k1 = (v*sin(p)*cos(t) + f*sin(pan)*sin(t)) / (-f*sin(t)*cos(pan) - v*cos(t)*cos(pan))
	b1 = (v * (-sin(t)*z + hc*sin(t)) + f*cos(t)*z - f*hc*cos(t)) / (-f*sin(t)*cos(pan) - v*cos(t)*cos(pan))
	w[0] = (b0 - b1) / (k1 - k0)
	w[1] = k0 * w[0] + b0

def box3D_construct(v_prms, img_shape, vp_line, f, t, hc):
	"""构造3D包络盒(3DBoxConstruct),输出包括8点三维坐标点及8点二维投影点(wTg3D, pTg2D)
	* v_W:输出的8个物理坐标点; v_P:输出的8个对应投影点;
	* v_prms:6个3D几何及约束参数; f, T, hc(height of camera):标定参数"""
	wTg3D, pTg2D = [], []  # 记录车辆3D包络的8个物理坐标点和对应的像素坐标点, 注：元素都是np格式
	w, l, h, u, v, vp_u = v_prms # 几何及约束参数赋值
	vp_v = vp_line[0] * (vp_u - vp_line[1]) + vp_line[2] # 求取灭点
	# 求取锚点(右下角)和约束灭点物理坐标
	w_vp1, w_orgin = np.zeros(3), np.zeros(3)
	w_orgin = uvTo_xyz(w_orgin, np.array([u, v]) - img_shape/2, f, t, hc)
	w_vp1 = uvTo_xyz(w_vp1, np.array([vp_u, vp_v]) - img_shape/2, f, t, hc)

	# 求取3D包络8点世界坐标
	k0 = (w_vp1[1] - w_orgin[1]) / (w_vp1[0] - w_orgin[0]) # 道路方向斜率
	k1 = -1.0 / k0  #垂直道路方向
	dev_x0 = l / sqrt(1 + k0 * k0)  # 车长方向向量
	dev_y0 = k0 * dev_x0
	dev_x1 = w / sqrt(1 + k1 * k1)  # 车宽方向向量
	dev_y1 = k1 * dev_x1
	wTg3D.append(w_orgin)  # p1入栈
	wTg3D.append(w_orgin-np.array([dev_x1, dev_y1, 0.0]))  # p2入栈
	wTg3D.append(wTg3D[1]-np.array([dev_x0, dev_y0, 0.0]))  # p3入栈
	wTg3D.append(w_orgin-np.array([dev_x0, dev_y0, 0.0]))  # p4入栈
	wTg3D.append(w_orgin-np.array([0.0, 0.0, -h]))  # p5入栈
	wTg3D.append(wTg3D[1]-np.array([0.0, 0.0, -h]))  # p6入栈
	wTg3D.append(wTg3D[2]-np.array([0.0, 0.0, -h]))  # p7入栈
	wTg3D.append(wTg3D[3]-np.array([0.0, 0.0, -h]))  # p8入栈

	# 反求3D包络8个点对应图像中的2D坐标
	for i in range(8):
		pTg2D.append(xyzTo_uv(wTg3D[i], f, t, hc) + img_shape/2)

	return wTg3D, pTg2D


