from tsgr.inference import TSGRFImageRecognizer

with TSGRFImageRecognizer(
    mediapipe_model="models/hand_landmarker.task",
    routing_model_root="results/fixed_routing_models",
    acceptance_root="results/acceptance",
    scenario="S1_ALL_IN_DOMAIN",
    fold_id="all",
) as recognizer:
    result = recognizer.predict("example.jpg")
    print(result.label)
