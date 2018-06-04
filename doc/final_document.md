# 车辆检测及型号识别

### 项目安装说明

以 Ubuntu 系统（LinuxMint）为例：

依次安装 Anaconda / TensorFlow / opencv-python 模块。

**安装 Anaconda：**

1. 打开 Ubuntu 系统，打开终端

2. 输入命令（注意 sh 后路径要改为 Anaconda 安装包的路径）

   $ sh Downloads/Anaconda3-5.1.0-Linux-x86_64.sh

   然后确认回车执行。

<img src="./reference image/conda_install2.png">

3. 然后会显示软件包的安装说明，这个时候按 q 键退出浏览就行，不然按回车要看很长时间。

<img src="./reference image/conda_install3.png">

4. 然后按照提示，输入 yes

<img src="./reference image/conda_install4.png">

5. 这个时候系统会提示你确认安装位置，默认按回车。也可以自己指定安装路径。回车选择默

   认位置就好。

<img src="./reference image/conda_install5.png">

6. 这里出现 warnning，提示要把 anaconda 的路径添加到启动文件里，默认 yes

<img src="./reference image/conda_install6.png">

7. 安装完成后,更新一下 bash

   $ source .bashrc

8. 输入

   $ python

   看到下面有 anaconda 的提示，就说明成功了。输入 quit() 退出 Python

<img src="./reference image/conda_install9.png">

**安装 TensorFlow：**

9. 下面来安装 Tensorflow，这里我们根据官网提示，用 conda 管理安装包。

   $ conda create -n tensorflow
   之后按照提示输入 y，就可以顺利创建环境了

<img src="./reference image/conda_install12.png">

10. 然后激活该环境

    $ source activate tensorflow

<img src="./reference image/conda_install13.png">

11. 然后更新一下 pip

    $ pip install --upgrade pip

<img src="./reference image/tf_install1.png">

12. 因为我们之前已经把 Tensorflow 下载到本地了，所以接下来要这样输入（注意最后路径改为本地 TensorFlow 安装包的路径）

    $ pip install --ignore-installed --upgrade Downloads/tensorflow-1.5.1-cp36-cp36m-linux_x86_64.whl

<img src="./reference image/tf_install2.png">

​	然后就是等待。。。。

13. 安装完成后不要退出,检验一下:
    $ python
    然后
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    看到 Hello, TensorFlow! 就说明安装成功。输入 quit() 退出 Python

<img src="./reference image/tf_install3.png">

**安装 opencv-python：**

14. 输入

    $ pip install opencv-python

    然后回车等待安装完毕

<img src="./reference image/cv2_install.png">

15. 下次再用的时候，记得先启动环境，再进 Python

    $ source activate tensorflow
    $ python

**项目代码拷贝：**

如图所示，将项目代码拷贝到某个目录（建议 ~/workspace），其中：

outputs -- 保存 inference 得到的绘有 bounding box 及其分类结果和得分的图片

uploads -- 保存上传的要进行 inference 的图片

faster_rcnn_resnet101_lowproposals_coco_2017_11_08.pb -- 用来进行检测 inference 的模型

labels.txt -- 车辆型号分类索引及名字对照

pj_vehicle_inception_v4_freeze.pb -- 用来进行分类 inference 的模型

server.py -- 主程序

server.sh -- 调用主程序的脚本

simhei.ttf -- 图片上写分类结果时用的字体（黑体，中文）

test.jpg -- 测试用图片例子

<img src="./reference image/code2copy.png">

### 项目使用说明

1. 打开 Ubuntu 系统，打开终端

2. 输入

   $ cd workspace/pj_vehicle

   回车进入项目代码所在文件夹（将代码拷贝到的目录里）

3. 输入

   $ sh server.sh

   回车执行脚本

<img src="./reference image/run_server_sh.png">

4. 在浏览器中打开终端中提示的网址：http://127.0.0.1:5001

<img src="./reference image/webpage.png">

5. 在网络上随意找一张包含汽车的图片，并存储到本地，记住存储位置

   点击网页中“Browse...”按钮在弹出窗口中选择刚才下载的图片，点击“Open”

<img src="./reference image/select_img.png">

6. 选择完以后，“Browse...”按钮右边会显示图片名，点击“上传”按钮，上传图片到 server 做 inference

<img src="./reference image/selected.png">

7. 耐心等待一会儿，页面就会返回 inference 的结果

<img src="./reference image/infer_result.png">

8. 可以继续选择图片上传做检测识别，或者关闭网页及终端结束。
9. 过程中绝对不可以先关闭终端；如果上传为空，网页会报错，可以重新打开网址恢复正常。
10. 结束可以在项目文件夹下的 uploads 文件夹找到原图，在 outputs 文件夹找到 inference 结果图片。

### 使用展示视频

见 "亓呈文 - ProjectShow.mp4"