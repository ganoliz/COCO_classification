# COCO_classification
Image classification of 4 classes(bird, airplane, train, boat) over [MS COCO 2017 datasets](https://cocodataset.org/#download).

![image](https://github.com/ganoliz/COCO_classification/blob/main/images/vis1.png)
![image](https://github.com/ganoliz/COCO_classification/blob/main/images/vis2.png)

## Usage

First, you can see my implementation in the demo folder [classification.ipynb](https://github.com/ganoliz/COCO_classification/blob/main/demo/classification.ipynb).
You can also test step-by-step on your own.

1. Download and place MS COCO data in the following path:

```bash
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip -d ./coco/images/
unzip val2017.zip -d ./coco/images/
unzip annotations_trainval2017.zip -d ./coco/
```
You should have disk capacity more than 50 GB (train2017.zip is 18 GB).

2. Download and place ours fine-tune [Resnet50 model](https://drive.google.com/file/d/10QcHIsOez4qxn4K93blD6f03GO_G8xSy/view?usp=sharing) checkpoint in the checkpoint folder.

3. Generate 4 classes annotation by running this script:

```bash
python annotation_prepare.py {args}
```

4. Visualize datasets

```bash
python vis_datasets.py {args}
```

5. Run evaluate.py scripts

```bash
python evaluate.py {args}
```

6. Visualize prediction 

```bash
python vis_prediction.py {args}
```

7. Try to train on your own.

Modify train.py code and run the script. 

```bash
python train.py {args}
```

## Training schema
We fine-tune a **ResNet-50** model for image classifcation on four distinct categories: bird, airplane, train and boat. The dataset undergoes **random resized cropping** and **horizontal flipping** to enhance generalization. Given the computational demands of training, we utilize an **NVIDIA A100 GPU** with a batch size of **2048**.
The model is optimized using the **Adam optimizer** with a learning rate of **1e-4** and **CrossEntropyLoss** serves as the loss function. To dynamically adjust learning rates, we employ a **ReduceLROnPlateau scheduler**, reducing the learning rate when validation performance plateaus.
Instead of retraining the entire network, we fine-tune only the **last two layers** (layer3 and layer 4) of ResNet-50, allowing the model to retain low-level feature representations while adapting high-level features to our specific task.
Training spans **70 epochs**, balancing sufficient learning without excessive overfitting. Throughout the process, performance is monitored using accuracy and confusion matrices, ensuring each class achieves a classification accuracy of at least 95 %.

## Results

Confusion Matrix:
```
[[49  2  0  0]
 [ 0 49  0  0]
 [ 0  1 48  0]
 [ 0  0  0 23]]
```
bird accuracy: 96.08%
airplane accuracy: 100.00%
train accuracy: 97.96%
boat accuracy: 100.00%

![image](https://github.com/ganoliz/COCO_classification/blob/master/images/test_example.png)