from ultralytics import YOLOv10
 
model = YOLOv10("ultralytics/cfg/models/v10/yolov10s.yaml")

model = YOLOv10('yolov10s.pt')
 
if __name__ == '__main__':
    model.train(data="nanhaifish/fold2.yaml",
                          epochs=100,
                          imgsz=1920,
                          batch=16,
                          device="0,1"
                          )
