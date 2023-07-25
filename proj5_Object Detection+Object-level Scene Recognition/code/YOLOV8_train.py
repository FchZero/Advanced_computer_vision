import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO


if __name__ == "__main__":
    # 训练
    # model = YOLO("yolov8n.yaml") # 从YAML构建一个新模型
    model = YOLO("yolov8n.pt")  # 加载预训练模型（从预训练模型开始训练）
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重
    # 训练模型
    model.train(data="coco128.yaml", epochs=100, imgsz=640)
    # 模型保存在你打开目录下的runs中

    # 验证
    # 训练后验证
    metrics = model.val()
    # 如果需要单独验证，可以使用下面的方法
    # 加载自定义模型
    # model = YOLO("runs/detect/train5/weights/best.pt")
    # 设置需要验证的数据
    # metrics = model.val(data="coco128.yaml")
    # 如果不设置数据的话，就使用model.pt中的data yaml文件
    # metrics = model.val()
