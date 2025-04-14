from ultralytics import YOLOv10

# Load a model
# model = YOLOv10('yolov10s.pt')  # load an official model
model = YOLOv10('runs/detect/train/weights/best.pt')  
metrics = model.val(data = 'nanhaifish/fold3.yaml',imgsz=1920,split='val')
inference_time = metrics.speed['inference']  
fps = 1000 / inference_time  

print(f"FPS: {fps:.2f} (平均推理时间)")
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

