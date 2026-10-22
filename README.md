不同型号无人机检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62b6346ca0354ea7a09b61915b0601b5.png)
，10000张，yolo，voc，coco三种标注方式
图像尺寸:627*940
类别数量:5类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a232fb64ec244f29f7361981c50142f.png)

训练集图像数量:7003; 验证集图像数量:2001； 测试集图像数量:996
类别名称: 每一类图像数 ，每一类标注数
mohajer_6 - 莫哈杰‑6 无人机：2788,3016
shahed_238 - 沙赫德‑238 无人机：2772,2995
mq9_reaper - MQ‑9 死神无人机：2805,3039
dji_mavic - 大疆御系列无人机：2736,2954
shahed_136 - 沙赫德‑136 无人机：2748,2962
image num: 10000![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f535ab8c549c41f79d6002b01d3dcf7e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5cb6796cbf8449db8fddf3c09b00bada.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a63a9be13b0448269a4b8ee02fe94252.png)
| 项目 | 详情 |
|---|---|
| 数据集名称 | 不同型号无人机检测数据集 |
| 总图像数量 | 10000 张 |
| 图像尺寸 | 627 × 940 |
| 标注格式 | YOLO、VOC、COCO 三种格式 |
| 类别数量 | 5 类 |
| 训练集图像数量 | 7003 张 |
| 验证集图像数量 | 2001 张 |
| 测试集图像数量 | 996 张 |

| 类别名称 | 图像数量 | 标注数量 |
|---|---:|---:|
| mohajer_6 - 莫哈杰-6 无人机 | 2788 | 3016 |
| shahed_238 - 沙赫德-238 无人机 | 2772 | 2995 |
| mq9_reaper - MQ-9 死神无人机 | 2805 | 3039 |
| dji_mavic - 大疆御系列无人机 | 2736 | 2954 |
| shahed_136 - 沙赫德-136 无人机 | 2748 | 2962 |

## YOLO 训练代码

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device=0,
    workers=2,
    project="drone_detection",
    name="train_result"
)

metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.4f}")

model.export(format="onnx")
```

## data.yaml

```yaml
path: ./drone_dataset
train: images/train
val: images/val
test: images/test

names:
  0: mohajer_6
  1: shahed_238
  2: mq9_reaper
  3: dji_mavic
  4: shahed_136
```
