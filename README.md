# ISL Gesture Recognition (A–Z)

This project detects static Indian Sign Language alphabet gestures (A-Z) using hand landmarks via MediaPipe and classifies them using an SVM model. It supports:

✅ Two-hand input  
✅ Word and sentence prediction  
✅ Real-time webcam input + voice output

## How to Run
1. Collect data → `collect_dataset_multi_hand.py`  
2. Merge → `merge_dataset.py`  
3. Train model → `train_model.py`  
4. Run real-time → `predict_realtime.py`
