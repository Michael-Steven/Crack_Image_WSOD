**Test** a PCL network. For example, test the VGG 16 network on VOC 2007:

#### On trainval
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
    --load_ckpt Outputs/vgg16_voc2007/$MODEL_PATH \
    --dataset voc2007trainval
  ```

#### On test
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
    --load_ckpt Outputs/vgg16_voc2007/$model_path \
    --dataset voc2007test
  ```

Test output is written underneath `$PCL_ROOT/Outputs`.

**If your testing speed is very slow (> 5 hours on VOC 2007), try to add `torch.backends.cudnn.enabled = False` after [this line of codes](https://github.com/ppengtang/pcl.pytorch/blob/master/tools/test_net.py#L119). See [issue #45](https://github.com/ppengtang/pcl.pytorch/issues/45#issuecomment-759160617) for more details.**

~~**Note: Add `--multi-gpu-testing` if multiple gpus are available.**~~

#### Evaluation
For mAP, run the python code tools/reval.py
  ```Shell
  python tools/reeval.py --result_path $output_dir/detections.pkl \
    --dataset voc2007test --cfg configs/baselines/vgg16_voc2007.yaml
  ```

For CorLoc, run the python code tools/reval.py
  ```Shell
  python tools/reeval.py --result_path $output_dir/discovery.pkl \
    --dataset voc2007trainval --cfg configs/baselines/vgg16_voc2007.yaml
  ```
