import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2


if __name__ == "__main__":
    # 加载模型
    # 加载官方模型
    # model = YOLO("yolov8n.pt")
    # 加载自定义模型
    model = YOLO("runs/detect/train5/weights/best.pt")

    # 预测
    # 图片预测
    # 不带参数预测
    # results = model("datasets/coco128/test")
    # 带参数预测
    results = model.predict(source="../datasets/coco128/test", save=True)
    # 默认保存在runs中
    # 绘制结果
    # results_plotted = results.plot()
    # cv2.imshow("results", results_plotted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 视频预测
    # 可以通过使用stream，可以创建结果对象的生成器，以减少内存使用。
    # results = model(source="A.mp4", show=True)
