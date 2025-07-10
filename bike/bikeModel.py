from ultralytics import YOLO
import os

def main():
    # === 1. SETUP PATHS ===
    DATA_YAML_PATH = r"D:\Full Bike.v4i.yolov8\data.yaml"  # <- No space at end
    MODEL_TYPE = "yolov8m.pt"
    EXPERIMENT_NAME = "bike_detector_m"
    PROJECT_DIR = "runs/detect"

    # === 2. TRAIN THE MODEL ===
    print("🚀 Starting training...")
    model = YOLO(MODEL_TYPE)

    model.train(
        data=DATA_YAML_PATH,
        epochs=120,
        imgsz=640,
        batch=8,
        device=0,
        name=EXPERIMENT_NAME,
        project=PROJECT_DIR,
        patience=20,
        verbose=True
    )

    # === 3. EVALUATE BEST MODEL ===
    print("\n📊 Validating best model for accuracy...")
    best_model_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "best.pt")
    model = YOLO(best_model_path)
    metrics = model.val()
    print(metrics)

    # === 4. INFERENCE ON NEW IMAGE ===
    print("\n🔍 Running inference...")
    TEST_IMAGE = r"D:\test.jpg"
    predict_results = model.predict(
        source=TEST_IMAGE,
        conf=0.3,
        save=True
    )

if __name__ == '__main__':
    main()
