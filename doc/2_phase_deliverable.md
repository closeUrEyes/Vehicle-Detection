# 第二周阶段成果

GitHub 上建立项目，并用 git 进行管理。

<img src="./reference image/GitHub.png"/>

在 Tinymind 上建立模型，代码使用 GitHub 项目代码。

<img src="./reference image/tinymind_pj.png"/>

上传数据集（训练集、校验集），并在模型中使用。InceptionV4 预训练模型采用第七周作业中的，训练代码也在第七周代码基础上稍作修改。

<img src="./reference image/dataset.png"/>

## 2.1 训练数据处理

object detection 并不需要对图像进行预处理，而 InceptionV4 网络的图像输入标准大小是 299×299，需要对输入图像进行 resize。另外 Inception 网络预处理部分会对图像随机施加一些变换如水平翻转、颜色变换等，以增加模型的鲁棒性。对 InceptionV4 网络训练模型在验证集上评估时，对图像做了中心裁切（Central Crop）变换。

## 2.2 Finetune 模型训练

训练过程在 TinyMind 上进行，见：https://www.tinymind.com/qichengwen/Vehicle-Detection/executions

- 训练过程出现的问题以及解决

  a. 查找 TFRecord 文件名匹配格式出错（未改）

<img src="./reference image/err_dataset_name.png"/>

​	b. 在 CPU 上训练时，clone_on_cpu 未改报错

<img src="./reference image/CPU_clone.png"/>

​	c. corrupted record 错误。提示数据丢失，发生于训练及验证过程中，而且 at 位置一直是同一个编号。用数据探索部分的 notebook 打开各个 TFRecord，随机读取 10 张图片，并未出错。考虑可能上传过程中数据部分缺失，重新上传了所有数据，仍然报错。搜索得到在读取图片时要做局部变量初始化，加入 tf.local_variable_initializer()，仍然未能解决问题。最终删除数据集，重新建立数据集，重新下载并上传数据，终于可以走完一个 epoch。问题原因很可能是第一次下载数据有中断出错。

<img src="./reference image/corrupted_error.jpg" />

​	d. 直接删除数据集，忘记在模型中先删掉，重新建立数据集后，即使与原数据集名字一致也不能识别。重新建立模型解决。

<img src="./reference image/warning1.png" />

- 训练过程可视化

  Finetune 训练了 15 个 epoch，最终得到验证集上准确率为 0.830078125 即 83.0%，top-5 召回率为 0.931640625 即 93.1%。

  训练过程 32 张图为 1 个 batch，每 10 个 step 输出当前 loss，训练完 1 个 epoch，在校验集上进行验证，输出准确率及 top-5 召回率。每 10 min 记录一次 summary，每 10 min 存一下 checkpoint。

  <img src="./reference image/training.png" >

  以下为每个 epoch 训练结束后，在验证集上的准确率（Accuracy）及 top-5 召回率（Recall_5）。准确率最终收敛于 0.83 左右。

  <img src="./reference image/epoch1.png" >

  <img src="./reference image/epoch2.png" >

  <img src="./reference image/epoch3.png" >

  <img src="./reference image/epoch4.png" >

  <img src="./reference image/epoch5.png" >

  <img src="./reference image/epoch6.png" >

  <img src="./reference image/epoch7.png" >

  <img src="./reference image/epoch8.png" >

  <img src="./reference image/epoch9.png" >

  <img src="./reference image/epoch10.png" >

  <img src="./reference image/epoch11.png" >

  <img src="./reference image/epoch12.png" >

  <img src="./reference image/epoch13.png" >

  <img src="./reference image/epoch14.png" >

  <img src="./reference image/epoch15.png" >

  - loss 可视化

  训练过程中依据 loss 变化，及时调整学习率，以 lr=0.01 训练了 2 个 epoch，以 lr=0.005 训练了 2 个 epoch，以 lr=0.001 训练了 6 个 epoch，以 lr=0.0005 训练了 2 个 epoch，以 lr=0.0001 训练了 3 个epoch。最后 loss 基本收敛，在验证集上的准确率也基本保持不变。

  <img src="./reference image/loss.png" >

## 2.3 可用系统的搭建

- 系统可以运行起来

首先，从 InceptionV4 Finetune 训练结束的 checkpoint，冻结参数导出模型 pj_vehicle_inception_v4_freeze.pb。

```
python -u export_inference_graph.py \
    --model_name=inception_v4 \
    --output_file=./pj_vehicle_inception_v4.pb \
    --dataset_name=pj_vehicle \
    --dataset_dir=../data/dataset

python -u ~/anaconda3/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
    --input_graph=pj_vehicle_inception_v4.pb \
    --input_checkpoint=./ckpt/model.ckpt-20610 \
    --output_graph=./pj_vehicle_inception_v4_freeze.pb \
    --input_binary=True \
    --output_node_name=InceptionV4/Logits/Predictions
```

然后以官方提供的 notebook 以及代码为基础，建立两个 notebook 分别探索分类（classification.ipynb）和物体检测（object_detection 目录中 object_detection.ipynb）两个部分的代码。

在测试中特意挑选了一张没有车的图片也进行测试。

**分类部分：**

分类结果取 predictions 里面取值最大的元素的索引，得分即最大值。

对没有车的图片进行测试，结果得分还是达到了18.9%，因此考虑在 object_detection 环节先对图片里有没有车做判断，没有就不再进行分类。

<img src="./reference image/cls_on_nocar.png" >

对与样本类似的正常有车图片，分类结果还比较满意。

<img src="./reference image/test.jpg" >

<img src="./reference image/cls_on_test.png" >

**检测部分：**

官方原始 notebook 得到结果如下，我们需要提取其中 car 分类并且得分最高的部分。

<img src="./reference image/out1.jpg" >

并且得到 squeeze 之后 boxes、classes、scores 的 shape 分别为 (300, 4) (300,) (300,)。

car 分类在 COCO 数据集分类中编号为 3。因此需要找出分类结果为 3 的 bounding-box 及其得分 score。 由下图可以看出，结果中分类为 car 的 boxes 正是按其得分 score 排序的。所以取分类为 car 的 boxes 中的第一个及其得分即可。

<img src="./reference image/boxes and scores.png" >

取出 box 的坐标如下图所示，形式为 box 左上角及右下角坐标 (y1, x1, y2, x2)，坐标经过了归一化。

<img src="./reference image/find_box.png" >

然后修改绘图传入的各个参数，只保留分类为 car 的 box 及其得分 score，将分类对应名字改为从 labels.txt 中查找的车型，分类结果采用分类部分得到的预测分类索引。

<img src="./reference image/out_prim.jpg" >

- 能够看到输入输出

上面按分类和检测两部分得到了可视化的结果，但是绘图函数中对中文的支持不好（不用 utf-8 编码会报 'latin-1' 编码错误），考虑之后自己重写这一部分。另外，对于图片中存在多辆车的情况，考虑按得到的 bounding box 对原始图片进行裁切，得到单纯的车的部分再送入分类模型做分类。