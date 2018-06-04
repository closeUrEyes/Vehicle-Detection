# 第三周阶段成果

## 模型训练完成

分类模型的 Finetune 上周已经完成，并也已导出可用于实际的冻结参数的 .pb 模型。按两个模型并行得出了初步的可视化结果，但图片中车型中文名称转成了 utf-8 编码，需要重写绘图部分代码。另外分类部分输入也改为用 bounding box 裁切后的部分。

### 结果可视化

建立了一个 crop and draw.ipynb 尝试对原始图片进行裁切，以及在原始图片上画 bounding box 及写分类结果和得分。

object detection 模型 inference 的结果已经得到分类为 car 的 bounding box，不过是归一化后的 box 左上角和右下角点坐标。乘以图片的 width 或 height 即得到绝对像素坐标。裁切使用 tf.image.crop_to_bounding_box() 函数。

<img src="./reference image/crop_code.png">

按 bounding box 裁切出来的部分如图所示，经过预处理之后输入 InceptionV4 模型得到最终分类结果及得分。

<img src="./reference image/crop.jpg">

分类结果及得分会以“车型名字 (score: 概率百分数%)”的形式在屏幕上打印。

b-box 裁切图片预处理部分采用训练模型时验证集上的预处理函数，包含中央裁切及双线性 resize 到 299×299。

<img src="./reference image/preproc_for_eval.png">

b-box 绘制及“写”分类结果采用 PIL 里面的 ImageDraw 模块，用 ImageFont 模块设置字体。用 draw.rectangle() 绘制出的 box 边框太细，所以采用了画边框四条线的方式来画 b-box。字体背景图块，高度等于字号如 18，长度等于文本用 gb2312 编码的长度乘以字号的一半。为了防止元素之间遮挡，按字体背景》文字》b-box 的顺序绘制。图片中文字是加在 b-box 框内的。

<img src="./reference image/draw_box_and_write_text.png">

### 效果分析

从网站上下载了一张与数据集中样本类似的图片，送入模型进行预测得到结果如下。车型识别准确，得分较高，bounding box 也基本正好框住了车体外形。整体来说，效果还不错。

<img src="./reference image/out.jpg">

## 系统搭建完成

开始项目计划采用最简单的命令行指定参数运行的方式。做的过程中感觉这种方式太不易用，于是仿照网上基于 Flask 的 webserver 方式以网页的形式进行交互。

另外，完善了程序中对没有车的图片的推断，根据其 object detection 中第 1 个分类为 car 的 bounding box 的得分如果低于 0.2 则停止后续的分类推断，直接输出“There's no car!” 的提示信息（将分类名字定为“There's no car!” ，得分定为 0.0%）。将 uploads 中上传的原图拷贝到 outputs 文件夹作为结果展示表示图上没有任何车。

```python
    else:
        # confidence level <= 0.2, means "there's no car"
        prob = 0.0
        class_name = 'There\'s no car!'
    
    if score > 0.2:
        # if there's any car, show the result image on the webpage
        new_url = '/static/%s' % os.path.basename(filename)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
    else:
        # if there's no car, show the primary image
        copy2(filename, os.path.join(FLAGS.output_folder, os.path.basename(filename)))
        new_url = '/static/%s' % os.path.basename(filename)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
```

两个文件夹一个 uploads 文件夹存上传的图片，一个 outputs 文件夹存推断的结果。

选择文件点击”上传“后产生一个 post 请求，将上传的文件存入 uploads 文件夹，并对其进行 inference，得到的结果图片存入 outputs 文件夹并在网页上展示，以及给出识别结果文字信息。

```python
@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
        <!doctype html>
        <title>实战车辆检测及型号识别</title>
        <h1>请 feed 一张图片</h1>
        <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file value='选择图片'>
            <input type=submit value='上传'>
        </form>
        <p>%s</p>
        """ % "<br>"
    # if post request, upload the image, and make inference
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            type_name = 'N/A'
            print('file saved to %s' % file_path)
            start_time = time.time()
            out_html = inference(file_path, 
                                 od_model_path=FLAGS.detection_model_path, 
                                 cls_model_path=FLAGS.class_model_path)
            # inference cost time
            duration = time.time() - start_time
            print('duration:[%.0fms]' % (duration*1000))
            return result + out_html 
    return result
```

### 能运行并根据合理的输入给出合理的输出

如图所示，首先 cd 进入脚本所在文件夹，然后 sh 执行 server.sh 脚本， 在浏览器打开提示的 http://127.0.0.1:5001/ 链接。

<img src="./reference image/run_sh.png">

<img src="./reference image/page1.png">

点击网页中“Choose File”选择一张图片然后点击”上传“，等待一下然后就会显示识别结果。

<img src="./reference image/page2.png">

### 没有明显不合理的设计

系统设计还是比较简单实用。

#### 输入输出可操作

输入一张图片上传做 inference 后，可以继续选择图片，输出的图片和文字提示可以保存或者从输出文件夹获取。

#### 对各种异常能够处理，系统不会崩溃

系统对上传文件的扩展名进行了限制（.jpg .jpeg .png）等等保证系统不会崩溃。对没有车的情况专门处理保证不会报错。