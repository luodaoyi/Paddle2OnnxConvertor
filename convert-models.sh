#!/bin/bash

function trainToOnnx() {
  python3 tools/export_custom.py --height ${1} \
  --width ${2} \
  -c ${3} \
  -o Global.pretrained_model=inference/${4}_train/best_accuracy \
  Global.load_static_weights=False \
  Global.save_inference_dir=inference/${4}_infer/

  paddle2onnx --model_dir inference/${4}_infer  \
   --model_filename  inference.pdmodel \
   --params_filename inference.pdiparams \
   --save_file inference/${4}_infer.onnx \
   --opset_version 11 \
   --enable_onnx_checker True
}


# ch_PP-OCRv2_rec
trainToOnnx 32 -1 configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml ch_PP-OCRv2_rec

# ch_PP-OCRv2_rec_slim_quant
trainToOnnx 32 -1 configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml ch_PP-OCRv2_rec_slim_quant

# ch_PP-OCRv2_det_distill
trainToOnnx -1 -1 configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml ch_PP-OCRv2_det_distill

# ch_ppocr_mobile_v2.0_cls
trainToOnnx 48 192 configs/cls/cls_mv3.yml ch_ppocr_mobile_v2.0_cls

# ch_ppocr_mobile_v2.0_det
trainToOnnx -1 -1 configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml ch_ppocr_mobile_v2.0_det

# ch_ppocr_server_v2.0_det
trainToOnnx -1 -1 configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml ch_ppocr_server_v2.0_det

# ch_ppocr_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml ch_ppocr_mobile_v2.0_rec

# ch_ppocr_server_v2.0_rec
trainToOnnx 32 -1 configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml ch_ppocr_server_v2.0_rec

# en_number_mobile_v2.0_rec
#trainToOnnx 32 -1 configs/rec/multi_language/rec_en_number_lite_train.yml en_number_mobile_v2.0_rec

# chinese_cht_mobile_v2.0_rec
#trainToOnnx 32 -1 configs/rec/multi_language/rec_ch_tra_lite_train.yml ch_tra_mobile_v2.0_rec


# rs_mobile_v2.0_rec
#trainToOnnx 32 -1 configs/rec/multi_language/rec_rs_cyrillic_lite_train.yml rs_cyrillic_mobile_v2.0_rec
#trainToOnnx 32 -1 configs/rec/multi_language/rec_rs_latin_lite_train.yml rs_latin_mobile_v2.0_rec
