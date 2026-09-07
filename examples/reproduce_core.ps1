$DATASET = ".\data\tsgr_dataset"
$RESULTS = ".\results"
$MODEL = ".\models\hand_landmarker.task"
$BRANCH = "raw.wrist_middle_mcp"

# Validate the released dataset before creating derived artifacts.
tsgr-audit-dataset $DATASET --require-videos --strict

# Process training photographs and both test perspectives.
tsgr-process-training $DATASET --model $MODEL --all-data --detection-profile high_recall --report-name reference_training --workers 16 --progress

tsgr-process-testing $DATASET --model $MODEL --test-mode image --handedness-policy ignore_handedness --all-data --minimum-success-rate 0.95 --detection-profile high_recall --no-video-recovery --report-name reference_image --workers 16 --progress --no-fail-fast

tsgr-process-testing $DATASET --model $MODEL --test-mode video --handedness-policy ignore_handedness --all-data --fallback-fps 30 --minimum-success-rate 0.95 --detection-profile balanced --video-recovery --video-recovery-after 1 --report-name reference_video --workers 16 --progress --no-fail-fast

# Build the evaluation reference and experiment folds.
tsgr-build-hand-presence-reference $DATASET --model $MODEL --output-dir "$RESULTS\ground_truth\hand_presence" --workers 16 --progress

tsgr-analyze-test-ground-truth $DATASET --hand-presence-reference "$RESULTS\ground_truth\hand_presence" --output-dir "$RESULTS\ground_truth\final"

tsgr-generate-folds $DATASET --output-dir "$RESULTS\experiment_plan" --workers 8 --progress

# Build fold-local reference models from TRAIN, then the fixed routing layer.
tsgr-build-fold-models "$RESULTS\experiment_plan" --all-scenarios --all-folds --branch $BRANCH --compact-correlation-threshold 0.995 --workers 4 --progress

tsgr-build-fixed-routing "$RESULTS\experiment_plan" --dataset-root-override $DATASET --output-dir "$RESULTS\fixed_routing_models" --all-scenarios --branch $BRANCH --feature-set compact --orientation-mode camera_aware --os-alpha 0.65 --c-top-n 5 --progress

# Continue with acceptance and held-out evaluation using the processing report
# directories created above. See README.md for the complete commands.
