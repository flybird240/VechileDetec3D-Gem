# VechileDetect3D(Gem)

基于几何约束的车辆3D检测的实现（PyQt框架）

1）代码逻辑关系思维导图：

2）相机标定操作逻辑流程图：

## 参考论文

1）入门论文：A Taxonomy and Analysis of Camera Calibration Methods for Traffic Monitoring Applications (https://pan.baidu.com/s/1M_4L4Iyj6ah0zhA14XzDuw?pwd=2023) 
提取码：2023)

2）2019年发表的优化标定论文：道路场景下相机 _王伟 (https://pan.baidu.com/s/1BU1jHNcU-_EBCWp7z2ySfQ?pwd=2023 
提取码：2023)

3）2022年发表的优化标定论文：Novel Optimization Approach for Camera Calibration (https://pan.baidu.com/s/1VRijnBvetEHRt3sKdm86zQ?pwd=2023 
提取码：2023)

4）数据集相关论文：Comprehensive Data Set for Automatic Single Camera (https://pan.baidu.com/s/1bq7gK1tlGPyPXtb-6e9t8A?pwd=2023 
提取码：2023)

## 数据集下载

链接：https://pan.baidu.com/s/1mYu4soz_FGIcJFYAUN6hZA?pwd=2023，提取码：2023

注：数据集中有21个视角(7个场景)，下载下来后，放入traffic_scenes_dataset文件夹下！

## 系统功能

**一、交通场景标识提取**

    (1) 道路边缘/灭点提取：通过道路边缘可获得道路宽度标识，灭点由道路边缘交点获得(必要条件)。
	
	(2) 虚线标识提取：虚线选取尽量靠近道路边缘附近，与道路边缘平行。
	
	(3) 冗余标识提取：场景图象中已知长度的标识线，可以不在道路平面上，但要平行于道路平面且已知高度。
	
	(4) 冗余平行线提取：尽量选取平行的线段，长度不定，但要保证在道路平面。
	
	(5) 清空所有标识：点击清空当前场景所有标识，即清空mark.json文档。
	
	注：在qt上可以直接操作图像交互，点击左键选取点，点击右键清除点，点击键盘Enter保存标识提取结果。


**二：基于单灭点的相机基础标定**
   
    (1) VWH相机基础标定，前提已提取过道路边缘/灭点(输入道路物理宽度)，手动输入正确的相机高度。
	
	(2) VLH相机基础标定，前提已提取过道路边缘/灭点，虚线标识(输入虚线物理长度)，手动输入正确的相机高度。
	
	(3) VWL相机基础标定，前提已提取过道路边缘/灭点(输入道路物理宽度)，虚线标识(输入虚线物理长度)。
	
	(4) 可直观显示标定误差，图像标识中的绿色部分为正确，红色部分为误差。
	
	注：绿色部分为原手工标注的标识，红色部分为反算的误差。
	

**三：相机标定优化**
   
    (1) 基于自旋角，相机高度，注意，操作时，数值变化后点击键盘Enter键才能启动标定。
	
	(2) 基于Least_square的标定优化，自动先执行基础标定，再做标定优化，效果一般，速度快。
	
	(3) 基于PSO粒子群的标定优化，自动先执行基础标定，再做标定优化，效果较好，速度慢。
	
	注：需保证有充足的正确的冗余标识/平行线，越充足、分散效果越好！另外标定优化不包括自旋角优化，自旋角优化需手动调节(主要考虑其速度较慢)。标定结果保存在calib文件夹下CalibResult.json。
	

**四：交互测距**

    (1) 选择场景图片同文件夹下有calib文件夹，且至少有一个标定文件才可使得交互测距控件可用。
	
## 说明

1）标定场景图像需以文件夹放入至traffic_scenes_dataset文件夹下。

2）代码在选择的场景图片同文件夹下自动新建mark和calib文件夹，用来保存标识结果和标定结果。标识文件为mark.json，和scene_mark.jpg(直观显示标识)，
标定结果为CalibResult.json，CalibResult_OP_least_square.json和CalibResult_OP_pso.json，分别对应无优化标定和两类优化标定。


## 开发环境

- Windows 10，Python 3.9

-依赖库：详见requirements.txt
