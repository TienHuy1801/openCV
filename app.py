from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtGui, QtCore, QtWidgets

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk     
from tkinter.filedialog import askopenfilename
import sys
import time


class MainApp(QMainWindow):
	img = "global"
	def __init__(self):
		super(MainApp, self).__init__()
		loadUi("app.ui", self)
		self.setWindowTitle("CV")
		self.btn1.clicked.connect(lambda: self.cvResize(int(self.resize_width.toPlainText()), int(self.resize_height.toPlainText())))
		self.btn2.clicked.connect(lambda: self.cvChangeColor(cv2.COLOR_BGR2HSV))
		self.btn3.clicked.connect(lambda: self.cvBlur((7, 7)))
		self.btn4.clicked.connect(lambda: self.cvCrop(int(self.crop_left.toPlainText()), int(self.crop_right.toPlainText()), int(self.crop_bottom.toPlainText()), int(self.crop_top.toPlainText())))
		self.btn5.clicked.connect(lambda: self.cvTranslate(int(self.translate_x.toPlainText()), int(self.translate_y.toPlainText())))
		self.btn6.clicked.connect(lambda: self.cvRotate(int(self.rotate_angle.toPlainText())))
		self.btn7.clicked.connect(lambda: self.cvFlip(str(self.flip_code.currentText())))
		self.btn8.clicked.connect(self.cvInput)
		###################################
		self.btn9.clicked.connect(lambda: self.cvNormalize(int(self.normalize_min.toPlainText()), int(self.normalize_max.toPlainText())))
		self.btn10.clicked.connect(lambda: self.cvThresholding(int(self.threshold.toPlainText())))
		############################################
		self.btn11.clicked.connect(self.cvImageNagative)
		self.btn12.clicked.connect(self.logTransformations)
		self.btn13.clicked.connect(lambda: self.powerlawTransformations(int(self.gamma.toPlainText())))
		self.btn14.clicked.connect(lambda: self.piecewiseLinearTransformations(int(self.s1.toPlainText()), int(self.s2.toPlainText())))
		self.btn15.clicked.connect(self.perspectiveTransfomation)
		self.btn16.clicked.connect(self.thresholdingTransformations)
		self.btn17.clicked.connect(self.showHistograms)
		self.btn18.clicked.connect(self.showEqualHist)
		
###########################################################
	def cvInput(self):
		Tk().withdraw()
		filename = askopenfilename()
		self.img = cv2.imread(filename)
# 		self.img = cv2.imread("D:/Python/anh1.jpg")
		h, w, ch = self.img.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(self.img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
		self.image.setPixmap(QtGui.QPixmap.fromImage(convert_to_Qt_format))

	def cvResize(self, width, height):
		if (type(self.img) == str): 
			print("Pls input Image")
			return 
		cv2.imshow('image', cv2.resize(self.img, (width, height), interpolation = cv2.INTER_AREA))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def cvChangeColor(self, color):
		if (type(self.img) == str): 
			print("Pls input Image")
			return 
		cv2.imshow('image', cv2.cvtColor(self.img, color))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# (ksize = (x, y) x,y > 0)
	def cvBlur(self, ksize):
		if (type(self.img) == str): 
			print("Pls input Image")
			return 
		cv2.imshow('image', cv2.GaussianBlur(self.img, ksize, cv2.BORDER_DEFAULT))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def cvCrop(self, left, right, bottom, top):
		if (type(self.img) == str): 
			print("Pls input Image")
			return 
		im = self.img[left:right, bottom:top]
		cv2.imshow('image', self.img[left:right, bottom:top])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# -x: left, x: right
	# -y: up, y: down
	def cvTranslate(self, x, y):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		transMat = np.float32([[1,0,x], [0,1,y]])
		dimensions = (self.img.shape[1], self.img.shape[0])
		cv2.imshow('image', cv2.warpAffine(self.img, transMat, dimensions))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def cvRotate(self, angle, rotPoint=None):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		(height, width) = self.img.shape[:2]
		if rotPoint is None:
			rotPoint = (width//2, height//2)
		rotMat = cv2. getRotationMatrix2D(rotPoint, angle, 1.0)
		dimensions = (width, height)
		cv2.imshow('image', cv2.warpAffine(self.img, rotMat, dimensions))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	# 1: horizontally
	# 0: vertically
	# -1: both vertically and horizontally
	def cvFlip(self, flipCode):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		if (flipCode == "horizontally"): 
			code = 1
		if (flipCode == "vertically"): 
			code = 0
		if (flipCode == "both vertically and horizontally"): 
			code = -1
		cv2.imshow('image', cv2.flip(self.img, code))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

################################################################
	def cvNormalize(self, min0 = 0, max0 = 255):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		ar = np.array(self.img).astype(np.uint8)
		mn = np.min(ar)
		mx = np.max(ar)
		norm = (ar - mn) * ((max0 - min0) / (mx - mn)) + min0
		cv2.imshow('image', norm.astype(np.uint8)) 
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def cvThresholding(self, threshold = 128):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		norm = gray
		norm[gray < threshold] = 0
		norm[gray >= threshold] = 255
		cv2.imshow('image', norm)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
################################################################
	def cvImageNagative(self):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		cv2.imshow('image', 255 - self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def logTransformations(self):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		thresh = 1.55
		#tmpImg = np.uint8(np.log1p(input))
		#output = cv2.threshold(tmpImg, thresh, 255, cv2.THRESH_BINARY)[1]
		output = thresh * np.log(1 + img /255)
		cv2.imshow('image', output)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	def powerlawTransformations(self, gamma):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		output = np.power(img, gamma) 
		cv2.imshow('image', output)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def piecewiseLinearTransformations(self, s1, s2):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		height, width = self.img.shape[:2] 

		rMin = self.img.min()  #  The minimum value of the gray level of the original image
		rMax = self.img.max()  #  The maximum gray level of the original image
		#r1, s1 = rMin, 0  # (x1,y1)
		#r2, s2 = rMax, 255  # (x2,y2)
		r1 = rMin
		r2 = rMax

		if s1 < 0 or s1 > 255 or s2 < 0 or s2 > 255 or s1 > s2:
			s1 = 0
			s2 = 255

			
		imgStretch = np.empty((height,width), np.uint8)  #  Create a blank array
		k1 = s1 / r1  # imgGray[h,w] < r1:
		k2 = (s2 - s1) / (r2 - r1)  # r1 <= imgGray[h,w] <= r2
		k3 = (255 - s2) / (255 - r2)  # imgGray[h,w] > r2
		for h in range(height):
			for w in range(width):
				if self.img[h,w] < r1:
					imgStretch[h,w] = k1 * self.img[h,w]
				elif r1 <= self.img[h,w] <= r2:
						imgStretch[h,w] = k2 * (self.img[h,w] - r1) + s1
				elif self.img[h,w] > r2:
						imgStretch[h,w] = k3 * (self.img[h,w] - r2) + s2		
		plt.figure(figsize=(10,3.5))
		plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=0.1, hspace=0.1)
		plt.subplot(131), plt.title("s=T(r)")
		x = [0, r1, r2, 255]
		y = [0, s1, s2, 255]
		plt.plot(x, y)
		plt.axis([0,256,0,256])
		plt.text(105, 25, "(r1,s1)", fontsize=10)
		plt.text(120, 215, "(r2,s2)", fontsize=10)
		plt.xlabel("r, Input value")
		plt.ylabel("s, Output value")
		plt.subplot(132), plt.imshow(self.img, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
		plt.subplot(133), plt.imshow(imgStretch, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
		plt.show()
		# plt.savefig('foo.png')
		cv2.imshow('image', imgStretch)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
	
	def perspectiveTransfomation(self):
		if (type(self.img) == str): 
			print("Pls input Image")
			return
		img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = img.shape

		pt1 = np.float32([[56,65],[368,52],[28,387],[389,290]])
		pt2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

		matrix = cv2.getPerspectiveTransform(pt1, pt2)
		new_img = cv2.warpPerspective(img, matrix, (cols,rows))
		cv2.imshow('image', new_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
	
	def thresholdingTransformations(self):
		output = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		output[output < 210] = 0
		
		cv2.imshow('image', output)
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
	#########################
	def Hist(self, input):
		s = input.shape
		H = np.zeros(shape=(256,1))

		for i in range(s[0]):
			for j in range(s[1]):
				k = input[i,j]
				H[k,0] = H[k, 0] + 1
		return H

	def showHistograms(self):
		img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.convertScaleAbs(img_gray, alpha=1.10 , beta= -20)
		s = img_gray.shape
		histg = self.Hist(img_gray)
		plt.plot(histg)
		x = histg.reshape(1,256)
		y = np.array([])
		y = np.append(y, x[0,0])
		
		for i in range(255):
			k = x[0, i + 1] + y[i]
			y = np.append(y,k)
		y = np.round((y / s[0] * s[1]) * (256 - 1))
		
		#H = Hist(img_1)
		#plt.plot(H)
		plt.show()
	## Histogram Equalization ##
	def calcGrayHist(self, I):
		h, w = I.shape[:2]
		grayHist = np.zeros([256], np.uint64)
		for i in range(h):
			for j in range(w):
				grayHist[I[i][j]] += 1
		return grayHist

	def equalHist(self, img):
		h, w = img.shape[0], img.shape[1]
		# Tính độ nạp thẳng độ màu xám
		grayHist = self.calcGrayHist(img)
		# Tính toán sơ đồ hình vuông thang độ xám tích lũy
		zeroCumuMoment = np.zeros([256], np.uint32)
		for p in range(256):
			if p == 0:
				zeroCumuMoment[p] = grayHist[0]
			else:
				zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
		# Nhận mối quan hệ ánh xạ giữa mức màu xám đầu vào và mức màu xám đầu ra theo sự tích lũy của thang độ xám
		outPut_q = np.zeros([256], np.uint8)
		cofficient = 256.0 / (h * w)
		for p in range(256):
			q = cofficient * float(zeroCumuMoment[p]) - 1
			if q >= 0:
				outPut_q[p] = np.floor(q)
			else:
				outPut_q[p] = 0
		# Nhận hình ảnh sau khi bản đồ công thức cân bằng
		equalHistImage = np.zeros(img.shape, np.uint8)
		for i in range(h):
			for j in range(w):
				equalHistImage[i][j] = outPut_q[img[i][j]]

		return equalHistImage

	def showEqualHist(self):
		input = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		equa = self.equalHist(input)
		plt.imshow(equa)
		plt.show()
################################################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    mainApp = MainApp()
    widget.addWidget(mainApp)
    widget.setFixedWidth(1920)
    widget.setFixedHeight(1080)
    widget.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
