# RT-DETR (Tracker)
Ultralytics YOLOv8 Docs RT-DETR (Realtime Detection Transformer)
Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector
Overview
Real-Time Detection Transformer (RT-DETR), developed by Baidu, is a cutting-edge end-to-end object detector that provides real-time performance while maintaining high accuracy. It leverages the power of Vision Transformers (ViT) to efficiently process multiscale features by decoupling intra-scale interaction and cross-scale fusion. RT-DETR is highly adaptable, supporting flexible adjustment of inference speed using different decoder layers without retraining. The model excels on accelerated backends like CUDA with TensorRT, outperforming many other real-time object detectors.

<img src="https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png" alt="Project Image" width="1000"/>

<img src="demo.gif" width="1000"/> 

### Features:

Efficient Hybrid Encoder: Baidu's RT-DETR uses an efficient hybrid encoder that processes multiscale features by decoupling intra-scale interaction and cross-scale fusion. This unique Vision Transformers-based design reduces computational costs and allows for real-time object detection.
IoU-aware Query Selection: Baidu's RT-DETR improves object query initialization by utilizing IoU-aware query selection. This allows the model to focus on the most relevant objects in the scene, enhancing the detection accuracy.
Adaptable Inference Speed: Baidu's RT-DETR supports flexible adjustments of inference speed by using different decoder layers without the need for retraining. This adaptability facilitates practical application in various real-time object detection scenarios.

### Pre-trained Models
The Ultralytics Python API provides pre-trained PaddlePaddle RT-DETR models with different scales:

- RT-DETR-L: 53.0% AP on COCO val2017, 114 FPS on T4 GPU
- RT-DETR-X: 54.8% AP on COCO val2017, 74 FPS on T4 GPU


### Prerequisites

- Python 3.x
- OpenCV
- PyTorch
- NumPy

### Installation

1. Clone this repository.
2. Install the required dependencies

```bash
conda create -n test1 python=3.10 -y
conda activate test1
pip install torch ultralytics opencv numpy
```

### Usage

1. Provide the video path in the code.
2. set [track=False] if dont want to track objects else ignore..
3. Run the script.
4. View the results. 

For more detailed usage instructions and options, refer to the project documentation.

### Run

```bash
python3 demo.py
```

### demo

<div align="center">
    <a href="https://github.com/bharath5673/RT-DETR">
        <img src="output.gif" alt="RT-DETR" width="1000"/>
    </a>
    <p>
        demo
    </p>
</div>


### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


### Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT)
* [https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
* [https://opencv.org/](https://opencv.org/)
* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* [https://github.com/NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)
</details>
