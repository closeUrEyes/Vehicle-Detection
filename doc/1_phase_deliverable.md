# 第一周阶段成果

## 1.	构建问题，看看整个图景

### 1.1 定义商业目标

实现一个车辆检测的工业级系统。

### 1.2 解决方案将如何使用？

从任意图片网站上，随机下载一张有汽车在内的图片，送入系统进行检测。

可以输出并显示图片中车辆的位置和型号等信息。

没有车辆的图片可以给出没有检测到的提示。

### 1.3 你该如何构建这个问题？

监督学习（supervised learning）问题：训练数据有类别标签（label）及车辆位置标注。

离线（offline）学习：暂时只有这一批量训练数据。

基于模型的学习（model-based learning）。

分类及回归任务（classification & regression task）：既要给出车辆的型号，同时也要给出车辆的在图上的位置信息。

深度学习中物体检测（object detection）问题：Classification + Localization。

### 1.4 性能如何度量？

~~以测试集上车辆检测分类（Classification）和定位（Localization）两部分损失加权和作为整体损失度量。~~

~~$total\_loss = classification\_loss + \alpha·regression\_loss​$~~

~~其中 $\alpha$ 为惩罚因子。$classification\_loss$ 采用 Softmax 交叉熵损失，$regression\_loss$ 采用平滑 L1 损失。~~

测试集上型号识别的准确率，得到的 Bounding-box 与肉眼识别的外框符合程度。

还有每张图片做 inference 平均用时（毫秒）。

### 1.5 性能度量与商业目标一致吗？

一致。

### 1.6 达到商业目标所需的最低性能指标？

准确率高于 80%

Bounding-box 与肉眼识别的外框没有明显偏移

速度：10fps（100ms 一张）

### 1.7 有没有相似问题？可以重用经验或工具？

第 7 周 InceptionV4 Finetune 作业。

TensorFlow models/research/object_detection 里提到的物体检测 model zoo 预训练模型。

## 2.	获取数据

### 2.1 下载数据

已从提供的百度云网盘链接下载所有训练和测试数据（TFRecord 格式）。

共占用 5GB 存储空间。

### 2.2 数据格式的理解

TFRecord 数据文件是一种将二进制数据（如图像数据等）和标签（训练的类别标签）统一存储的二进制文件，能更好的利用内存，在 TensorFlow 中快速地复制，移动，读取，存储等。TFRecord 文件包含了 tf.train.Example 协议缓冲区（protocol buffer，协议缓冲区包含了特征 Features）。你可以写一段代码获取你的数据，将数据填入到 Example 协议缓冲区（protocol buffer），将协议缓冲区序列化为一个字符串，并且通过 tf.python_io.TFRecordWriter class 写入到 TFRecords 文件。

从 TFRecords 文件中读取数据， 可以使用 tf.TFRecordReader 的 tf.parse_single_example 解析器。这个操作可以将 Example 协议内存块（protocol buffer）解析为 Tensor。

从技术角度讲，TFRecord 文件是 protobuf 格式的文件。

## 3.	探索数据获得洞察

### 3.1 创建一个 Jupyter notebook 记录数据探索过程

```python
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import os

# 加载 TFRecord 文件
tf_record_filename_queue = tf.train.string_input_producer(
    tf.gfile.Glob(os.path.join("./validation_set/",'*.tfrecord')), shuffle=True)

# 注意这个不同的记录读取器，它的设计意图是能够使用可能会包含多个样本的 TFRecord 文件
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

# 标签和图像都按字节存储，但也可按 int64 或 float64 类型存储于序列化的 tf.Example protobuf 文件中
tf_record_features = tf.parse_single_example(
    tf_record_serialized, features={
        'image/encoded': tf.FixedLenFeature([], tf.string), 
        'image/width': tf.FixedLenFeature([], tf.int64), 
        'image/height': tf.FixedLenFeature([], tf.int64), 
        'image/format': tf.FixedLenFeature([], tf.string), 
        'image/class/label': tf.FixedLenFeature([], tf.int64), 
    })

# 使用 tf.uint8 类型，因为所有的通道信息都处于 0~255
tf_record_image = tf.image.decode_jpeg(tf_record_features['image/encoded'], channels=3)

tf_record_width = tf.cast(tf_record_features['image/width'], tf.int32)
tf_record_height = tf.cast(tf_record_features['image/height'], tf.int32)
tf_record_label = tf.cast(tf_record_features['image/class/label'], tf.int32)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 启动多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        plt.imshow(tf_record_image.eval())
        plt.show()
        single, lbl, width, height = sess.run([tf_record_image, tf_record_label, tf_record_width, tf_record_height])
        img = Image.fromarray(single, 'RGB')
        img.save('IMG' + str(i) + '_' + str(width) + 'x' + str(height) + '_Label_' + str(lbl) + '.jpg')
        
    coord.request_stop()
    coord.join(threads)
```

### 3.2 对于监督学习任务，识别 the target attribute（s）

TFRecord 中，'image/class/label' 即为图片的标签（型号）。

### 3.3 可视化数据

从验证集中随机抽取一张图片，434x320 大小，Label 编号 526（现代-索纳塔八）。

共随机抽取了 10 张图片，图片中心区域大部分都只有一辆车，但个别有多辆车。

<img src="./reference image/IMG5_434x320_Label_526.jpg" />

### 3.4 确认可能应用的变换

数据增广（augmentation）：裁切（平移）、水平翻转、（旋转、）饱和与平衡变换等。

## 4.	探索许多不同模型，初步选出最好的

### 4.1 COCO-trained models {#coco-models}

| Model name                                                   | Speed (ms) | COCO mAP | Outputs |
| ------------------------------------------------------------ | ---------- | -------- | ------- |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) | 30         | 21       | Boxes   |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) | 42         | 24       | Boxes   |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2017_11_08.tar.gz) | 58         | 28       | Boxes   |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2017_11_08.tar.gz) | 89         | 30       | Boxes   |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2017_11_08.tar.gz) | 64         |          | Boxes   |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2017_11_08.tar.gz) | 92         | 30       | Boxes   |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz) | 106        | 32       | Boxes   |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2017_11_08.tar.gz) | 82         |          | Boxes   |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08.tar.gz) | 620        | 37       | Boxes   |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2017_11_08.tar.gz) | 241        |          | Boxes   |
| [faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2017_11_08.tar.gz) | 1833       | 43       | Boxes   |
| [faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2017_11_08.tar.gz) | 540        |          | Boxes   |

上表是 TensorFlow detection model zoo 中，在 COCO 数据集上预训练的一系列检测模型的表现。Speed 为对一张 600×600 的图像做 inference 的运行时间（毫秒），计算资源为一块 Nvidia GeForce GTX TITAN X。COCO mAP 为在 COCO 验证集上的平均查准率。由于 Tinymind 仅支持到 TensorFlow 1.4，这些是 models r1.5分支下包含的物体检测模型及使用 TensorFlow 1.4 版本训练的结果。

考虑到速度保证在 100ms 以内，可选模型有以下几种：

- 基于 MobileNets V1 框架的 Single Shot multibox Detector（SSD）模型。
- 基于 Inception V2 框架的 Single Shot multibox Detector（SSD）模型。
- 基于 Inception v2 框架的 Faster R-CNN 模型。
- 基于 ResNet-50 框架的 Faster R-CNN 模型。
- 基于 ResNet-101 框架的全卷积网络（R-FCN）模型。

可以明显地看出，SSD（Single Shot multibox Detector） 检测方法在速度上的优势，尤其是基础的特征提取网络采用 Google 针对嵌入式设备提出的一种轻量级的深层神经网络 — MobileNets 时。而基于 Region Proposal 的 Faster R-CNN 和 R-FCN 方法在 mAP 上还是还是领先很多。

<img src="./reference image/tradeoff.png" />

在检测模型选择上，主要是做 Speed 与 mAP 之间的 tradeoff。从表中及 tradeoff 做专门比较的论文中可以看到 Resnet 101 + 100 proposal Faster RCNN 模型在速度与平均查准率之间取得了一个最好的平衡，在本项目中也推荐使用此方法（下载 faster_rcnn_resnet101_lowproposals_coco 预训练模型）。

### 4.2 基于 Region Proposal 的 R-CNN 系列物体检测方法

#### R-CNN

**A.** Selective Search 寻找可能包含物体的方框

**B.** 接 CNN 提取特征，然后做 SVM 分类

**C.** 和物体标注框的位置的回归来修正 Selective Search 提出的原始物体框

其中 Selective Search 采用的是超像素合并的思路。Bounding-box regressors 采用的是 Linear。

<img src="./reference image/R-CNN.png" />

#### Fast R-CNN

Selective Search 提出**上千个**框都要**单独**过一遍 CNN 进行前向计算

思路：将整张图片执行一次 CNN 前向计算，到最后一层特征图时，通过**某种方式**把目标物体所在区域部分的特征图拿出来作为特征给分类器和回归器。

**ROI Pooling：**直接把 ROI 划分为固定的大小（让输入神经网络的图像大小不再固定）。

ROI 提取特征后，把物体框的回归和分类这两个任务的 **loss 融合**在一起训练，相当于**端到端的多任务**训练（end-to-end with a multi-task loss）。

其中，在计算预测的框和标准框的 loss 时，采用了一种叫做 $Smooth_{L1}$ 的 loss 计算方法：

$L_{loc}(t^u, v) = \sum_{i\in\{x, y, w, h\}}smooth_{L1}(t_i^u - v_i)$

其中：

$smooth_{L1}(x) = \begin{cases}\begin{matrix}{0.5 x^2}, & if\;|x|<1 \\ |x|-0.5, & 其他 \end{matrix} \end{cases}$

小的偏差利用 L2 计算，大的偏差利用 L1 计算。$smooth_{L1}$ 对偏差很大的值没有那么敏感，提高了 loss 计算的稳定性。

ROI Pooling 之后用 **SVD 分解**然后忽略次要成分，减小了全连接层计算量。

<img src="./reference image/Fast R-CNN.png" />

<img src="./reference image/ROI Pooling.png" style="zoom:40%" />

#### Faster R-CNN

Selective Search 成了限制计算效率的瓶颈。考虑用神经网络的办法取代之。

Faster R-CNN 中 **Region Proposal Networks（RPN）**被提出来替代Selective Search。算法的所有步骤都被包含到一个完整的框架中，实现了端到端的训练。

思路：最后一层特征图中是可以包含粗略的位置信息的，所以 Region Proposal 完全可以放到最后一层特征图上来做。

做法：

**A.** 首先对基础网络的最后一层特征图，执行一次 n×n 卷积（3×3 Conv+ReLU），输出指定通道数（256）的特征图，这步相当于用滑窗法对特征图进行特征提取。

**B.** 然后对得到的特征图的每个像素分别进入两个全连接层（1×1 Conv+ReLU）：一个计算该像素对应位置是否有物体的分数，输出是或否的分数，所以有 2 个输出；另一个计算物体框的二维坐标和大小，所以有 4 个输出。

**C.** 对每个像素，都尝试用中心在该像素位置，不同大小和不同长宽比的窗口作为 anchor box，回归物体框坐标和大小的网络是在 anchor box 基础上做 offset。k 个 anchor box -> 是否有物体分数的输出有 2k 维/通道，计算物体框坐标和大小的输出有 4k 维/通道。（3 种尺寸和 3 种长宽比：k=9）

**D.** 经过 NMS 和分数从大到小排序筛选出有效的物体框，从中随机选取作为一个 batch。然后通过 ROI Pooling 进行分类的同时，会进一步对物体框的位置及大小进行回归。两个任务的 loss 放一起实现了端到端的训练。

<img src="./reference image/RPN.png" />

<img src="./reference image/Faster R-CNN.png" />

#### R-FCN

Region-based Fully Convolutional Net（基于区域建议的全卷积网络）

采用全卷积网络最大化参数共享的程度来提升速度。

将 Ground Truth 分成 3×3 的小方格，相应的 Region Proposal 也会分成 9 个区域，吸收了 DPM 想法。

添加一个全卷积层，以生成位置敏感分数图的 score bank。这里应该有 k(C+1) 个分数图，其中，k代表切分一个目标的相关位置的数量（比如，9 代表一个 3x3 的空间网格），C+1 代表 C 个类外加一个背景。

运行一个全卷积 region proposal 网络（RPN），以生成感兴趣区域（regions of interest，RoI）。

对于每个 RoI，我们都将其切分成同样的 k 个子区域，然后将这些子区域作为分数图。

对每个子区域，我们检查其  score  bank，以判断这个子区域是否匹配具体目标的对应位置。比如，如果我们处在「上-左」子区域，那我们就会获取与这个目标「上-左」子区域对应的分数图，并且在感兴趣区域（RoI  region）里对那些值取平均。对每个类我们都要进行这个过程。

一旦每个 k子区域都具备每个类的「目标匹配」值，那么我们就可以对这些子区域求平均值，得到每个类的分数。

通过对平均后得到的 C+1 个维度向量进行 softmax 回归，来对 RoI 进行分类。

R-FCN 能同时处理位置可变性（location variance，无论哪个位置都能准确分类）与位置不变性（location invariance，物体在哪就能在哪画框）。它给出不同的目标区域来处理位置可变性，让每个 region proposal 都参考同一个分数图 score bank 来处理位置不变性。

集大成作品。

<img src="./reference image/R-FCN_meta.png" style="zoom:60%"/>

<img src="./reference image/R-FCN.png" />

### 4.3 基于 Regression 的 Single Shot 系列物体检测方法

#### YOLO：You Look Only Once

**基本思想：**把一幅图片划分成 S×S 的格子，以每个格子所在位置和对应内容为基础，来预测：

​	1）物体框，包含物体框相对格子中心的坐标（x, y）和物体框的宽 w 和高 h。每个格子预测 B 个物体框。

​	2）每个物体框是否有物体的置信度。

​	3）每个格子预测一共 C 个类别的概率分数。

每个格子需要输出的信息维度是 B×(4+1)+C 

**实现：**一幅图片首先缩放为一个正方形的图片（448×448），然后送进一个 CNN，到最后一层特征图时，接两层全连接，输出（并 reshape）是 7×7×30（B=2，C=20）。最后从这 7×7×30 张量中提取出来的物体框和类别的预测信息经过 NMS，就得到了最终的物体检测结果。

<img src="./reference image/YOLO.png" />

#### SSD: Single Shot MultiBox Detector

同时借鉴 YOLO 和 Faster R-CNN。

与 YOLO 相近的是，在 CNN 最后阶段，得到 S×S 特征图。然后是和 Faster R-CNN 相近的地方，借鉴 anchor box 的思想基于每个格子的位置生成默认的物体框。

相比 YOLO，主要改进是从一个分辨率较大的特征图开始，逐渐得到分辨率更低的特征图，每个分辨率下的特征图都作为产生物体类别分数和物体框的格子。这样就得到了不同大小感受野对应的局部图像信息。

论文中，基于 VGG16 在 300×300 输入下，得到的 conv5 是 38×38 的特征图，每个像素上取 k=4，经过进一步降采样分别得到 19×19、10×10、5×5 的特征图，对这 3 个特征图取 k=6，最后继续降采样得到 3×3 和 1×1的特征图，取 k=4。 则每个类别一共得到 38×38×4+(19×19+10×10+5×5)×6+(3×3+1×1)×4=8732 个默认物体框。每个框用一个通道数为 (C+4)k 的卷积得到预测的框和结果。

比起 YOLO，输入分辨率更低，但感受野精细程度更高，而且默认物体框的数量高出两个数量级，结果就是执行速度和精度的双双提升。

<img src="./reference image/SSD anchor.png" />

<img src="./reference image/SSD.png" style="zoom:50%"/>



### 4.4 分类网络

#### Inception

Inception 模块的基本思想源于 NIN（Network In Network），把**卷积+激活**看作是一种广义线性模型（Generalized Linear Model），用更有效的结构代替。

- Inception v1 的网络，将 1x1，3x3，5x5 的 conv 和 3x3 的 pooling，stack 在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；采用 1x1 卷积核进行降维，降低了计算量，同时让信息通过更少的连接传递以达到更加稀疏的特性；

<img src="./reference image/inception_v1.png"/>

- v2 的网络在 v1 的基础上，进行了改进，一方面了加入了 BN 层，减少了 Internal  Covariate Shift（内部neuron 的数据分布发生变化），使每一层的输出都规范化到一个 N(0,  1) 的高斯，另外一方面学习 VGG 用 2 个 3x3 的 conv 替代 inception 模块中的 5x5，既降低了参数数量，也加速计算；

<img src="./reference image/inception_v2_conv.png" style="zoom:70%"/>

- v3 一个最重要的改进是分解（Factorization），将 7x7 分解成两个一维的卷积（1x7, 7x1），3x3 也是一样（1x3, 3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将 1 个 conv 拆成 2 个 conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从 224x224 变为了 299x299，更加精细设计了 35x35/17x17/8x8 的模块；

<img src="./reference image/inception_v3_conv.png" style="zoom:70%"/>

​	35 降维可以理解为 1+9+25（branch 结构，Filter Concat 拼合）

<img src="./reference image/inception_v3_conv35.png" style="zoom:70%"/>

- v4 研究了 Inception 模块结合 Residual  Connection 能不能有改进。发现 ResNet 的结构可以极大地加速训练，同时性能也有提升，得到一个 Inception-ResNet v2 网络，同时还设计了一个更深更优化的 Inception v4 模型，能达到与 Inception-ResNet v2 相媲美的性能。

<img src="./reference image/inception_v4.png" style="zoom:50%" />

#### ResNet

要解决的是训练网络的退化问题，即随着层数的加深到一定程度之后，越深的网络反而效果越差。

$y = H(x)$

$H(x) = F(x)+x$

$F(x) = H(x)-x$ 对应值残差，因而叫残差网络。

（如 a）数据经过了两条路线，一条是和一般网络类似的经过两个卷积层再传递到输出，另一条则是实现单位映射的直接连接的路线，称为 shortcut。如果前面层的参数已经到了一个很好的水平，那么再构建基本模块时，输入的信息通过 shortcut 得以一定程度的保留。

通过实验证明，这个模块很好地应对了退化问题，并且让可有效训练的网络层数更进一步。

（如 b）实际使用时，希望能够降低计算消耗，又进一步提出了“瓶颈（BottleNeck）”模块。先通过 1x1 卷积降维，然后正常 3x3 卷积，最后再 1x1 卷积将维度和 shortcut 对应上。

（如 c）再次改进，把 ReLU 移到 conv 层之前，相应地 shortcut 不再经过 ReLU，相当于输入输出直连（最根本的改进来源）。因为对于每个单元，激活函数到了仿射变换之前，所以论文中将这种改进叫做预激活残差单元（pre-activation residual unit）。

比较经典的 ResNet 一般是 3 中结构：即 50 层、101 层、152 层。

<img src="./reference image/resnet.png"/>

#### MobileNets

Google 针对手机等嵌入式设备提出的一种轻量级的深度神经网络，取名为MobileNets。

核心思想就是卷积核的巧妙分解，可以有效减少网络参数。

<img src="./reference image/mobilenets.jpg"/>

分解一个标准的卷积为一个 depthwise convolutions 和一个 pointwise convolution。简单理解就是矩阵的因式分解。

先进行空间的 3x3 卷积，再进行通道 1x1 的卷积。

<img src="./reference image/depthwise_separable_convolutions.jpg"/>

使用了大量的 3×3 的卷积核，极大地减少了计算量（1/8 到 1/9之间），同时准确率下降的很少，相比其他的方法确有优势。

## 5.	项目方案规划

### 5.1 模型规划

分类部分（车辆型号）采用第 7 章作业用到的 InceptionV4 网络。

定位部分（车辆位置）使用 faster_rcnn_resnet101_lowproposals_coco 预训练模型模型直接进行预测（物体检测部分采用 Faster R-CNN模型）。

定位会出现多个框，取其中车辆分类中最大的那个，其他忽略。

由于分类与定位网络是独立的，不能采用 end-to-end 也就是端到端的训练方式，采用分类与定位（不用训练）网络并行的方式。

### 5.2 系统规划

**系统架构**

<img src="./reference image/meta.png"/>

**输入：**采用命令行直接指定待识别文件的方式。

**输出：**将检测结果写入文件，并使用 matplotlib 显示检测结果。

**结果展示：**

屏幕打印型号及位置，型号不在范围内 -> “Unknown”，没有车打印“None”。并保存类似以下图片结果。

<img src="./reference image/output.png"/>