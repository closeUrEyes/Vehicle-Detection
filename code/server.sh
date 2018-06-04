python -u ./server.py --class_model_path=./pj_vehicle_inception_v4_freeze.pb \
    --detection_model_path=./faster_rcnn_resnet101_lowproposals_coco_2017_11_08.pb \
    --label_file=./labels.txt
