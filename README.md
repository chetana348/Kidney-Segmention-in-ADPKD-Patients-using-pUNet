# PCAnnUNet Medical Image Segmentation
Segments kidneys from ADPKD patient kidney images



### Features
-  3D data processing 
- Augmented patching technique, requires less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Sørensen and Jaccard similarity measurement as golden standard in medical image segmentation benchmarking
- Utilizing medical image headers to retrive space and orientation info after passthrough the network


## Usage
### Required Libraries
Known good dependencies
- Python 3.7
- Tensorflow 1.14 or above
- SimpleITK

### Folder Hierarchy
All training, testing and evaluation data should put in `./data`

    .
    ├── ...
    ├── data                      # All data
    │   ├── testing               # Put all testing data here
    |   |   ├── case1            
    |   |   |   ├── img.nii.gz    # Image for testing
    |   |   |   └── label.nii.gz  # Corresponding label for testing
    |   |   ├── case2
    |   |   ├──...
    │   ├── training              # Put all training data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   ├── img.nii.gz    # Image for training
    |   |   |   └── label.nii.gz  # Corresponding label for training
    |   |   ├── case2
    |   |   ├──...
    │   └── evaluation            # Put all evaluation data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   └── img.nii.gz    # Image for evaluation
    |   |   ├── case2
    |   |   ├──...
    ├── tmp
    |   ├── cktp                  # Tensorflow checkpoints
    |   └── log                   # Tensorboard logging folder
    ├── ...
    
If you wish to use image and label with filename other than `img.nii.gz` and `label.nii.gz`, please change the following values in `config.json`

```json
"ImageFilenames": ["img.nii.gz"],
"LabelFilename": "label.nii.gz"
```

In segmentation tasks, image and label are always in pair, missing either one would terminate the training process.

### Training

You may run train.py with commandline arguments. To check usage, type ```python main.py -h``` in terminal to list all possible training parameters.

Available training parameters
```console
  -h, --help            show this help message and exit
  -v, --verbose         Show verbose output
  -p [train evaluate], --phase [train evaluate]
                        Training phase (default= train)
  --config_json FILENAME
                        JSON file for model configuration
  --gpu GPU_IDs         Select GPU device(s) (default = 0)
 ```

The program will read the configuration from `config.json`. Modify the necessary hyperparameters to suit your dataset.

Note: You should always set label 0 as the first `SegmentationClasses` in `config.json`. Current model will only run properly with at least 2 classes.

#### Image batch preparation
Typically medical image is large in size when comparing with natural images (height x width x layers x modality), where number of layers could up to hundred or thousands of slices. Also medical images are not bounded to unsigned char pixel type but accepts short, double or even float pixel type. This will consume large amount of GPU memories, which is a great barrier limiting the application of neural network in medical field.

Here we introduce serveral data augmentation skills that allow users to normalize and resample medical images in 3D sense. In `train.py`, you can access to `trainTransforms`/`testTransforms`. For general purpose we combine the advantage of tensorflow dataset api and SimpleITK (SITK) image processing toolkit together. Following is the preprocessing pipeline in SITK side to facilitate image augmentation with limited available memories.

1. Image Normalization (fit to 0-255)
2. Isotropic Resampling (adjustable size, in mm)
3. Padding (allow input image batch smaller than network input size to be trained)
4. Random Crop (randomly select a zone in the 3D medical image in exact size as network input)
5. Gaussian Noise

The preprocessing pipeline can easily be adjusted with following example code in `./pipeline/pipeline2D.yaml`:
```yaml
train:
  # it is possible to first perform normalization on 3D image first, e.g. image normalization using statistical values 
  3D:

  2D:
    - name: "ManualNormalization"
      variables: 
        windowMin: 0
        windowMax: 600
    - name: "Resample"
      variables: 
        voxel_size: [0.75, 0.75]
    - name: "Padding"
      variables: 
        output_size: [256,256]
    - name: "RandomCrop"
      variables: 
        output_size: [256,256]
```

To write you own preprocessing pipeline, you need to modify the preprocessing classes in `./pipeline/NiftiDataset2D.py` or `./pipeline/NiftiDataset3D.py`.

Additional preprocessing classes (incomplete list, check `./pipeline/NiftiDataset2D.py` or `./pipeline/NiftiDataset3D.py` for full list):
- StatisticalNormalization
- Reorient (take care on the direction when performing evaluation)
- Invert
- ConfidenceCrop (for very small volume like cerebral microbleeds, alternative of RandomCrop)
- Deformations:
  The deformations are following SITK deep learning data augmentation documentations, will be expand soon.
  Now contains:
  - BSplineDeformation 

  **Hint: Directly apply deformation is slow. Instead you can first perform cropping with a larger than patch size region then with deformation, then crop to actual patch size. If you apply deformation to exact training size region, it will create black zone which may affect final training accuracy.**
  

```


### Evaluation
To evaluate image data, first place the data in folder ```./data/evaluate```. Each image data should be placed in separate folder as indicated in the folder hierarchy

There are several parameters you need to set in order manually in `EvaluationSetting` session of `./config/config.json`.

Run `main.py -p evaluate --config_json ./config/config.json` after you have modified the corresponding variables. All data in `./data/evaluate` will be iterated. Segmented label is named specified ini `LabelFilename` and output in same folder of the respective input image files.

Note that you should keep preprocessing pipeline similar to the one during training, but without random cropping and noise. You may take reference to `evaluate` session in `./pipeline/pipeline2D.yaml`.
#### Post Processing (To be updated)



## References:
- SimpleITK guide on deep learning data augmentation:
https://simpleitk.readthedocs.io/en/master/Documentation/docs/source/fundamentalConcepts.html
https://simpleitk.github.io/SPIE2018_COURSE/data_augmentation.pdf
https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf

