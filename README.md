# 🔬 Explainable AI for Medical Image Classification

A comprehensive deep learning project focused on explainable AI techniques for pneumonia detection from chest X-ray images. This project implements multiple state-of-the-art convolutional neural networks with explainability methods to provide transparent and interpretable AI predictions for medical diagnosis.

## ✨ Features

### Deep Learning Models
- **🤖 Ensemble Architecture**: Stacked ensemble combining MobileNetV2 and DenseNet169
- **🔄 Transfer Learning**: Pre-trained ImageNet weights for improved performance
- **📊 Multiple Architectures**: Support for various CNN architectures (DenseNet169, MobileNetV2, InceptionV3, Xception)
- **🎯 Binary Classification**: Pneumonia vs. Normal chest X-ray classification

### Explainability Methods
- **🔥 GradCAM (Gradient-weighted Class Activation Mapping)**: Visualize which regions of the image the model focuses on
- **📍 Class Activation Maps**: Heatmaps showing important regions for predictions
- **👁️ Visual Interpretability**: Overlay heatmaps on original images for intuitive understanding
- **📈 Model Transparency**: Understand model decision-making process

### Data Processing
- **🖼️ Image Preprocessing**: Resizing, normalization, and augmentation
- **📦 Data Augmentation**: Horizontal flips, rescaling for improved generalization
- **⚖️ Class Balancing**: Handling imbalanced datasets
- **🔄 Data Generators**: Efficient data loading with Keras ImageDataGenerator

### Model Training
- **📉 Early Stopping**: Prevent overfitting with validation monitoring
- **🎚️ Learning Rate Scheduling**: Adaptive learning rate reduction
- **💾 Model Checkpointing**: Save best models during training
- **📊 Training Metrics**: Track accuracy, loss, and validation performance

## 🛠️ Technologies Used

### Deep Learning Frameworks
- **TensorFlow/Keras**: Primary deep learning framework
- **TensorFlow 2.x**: Modern TensorFlow implementation

### Pre-trained Models
- **DenseNet169**: Densely Connected Convolutional Networks
- **MobileNetV2**: Lightweight mobile-optimized architecture
- **InceptionV3**: Inception architecture variant
- **Xception**: Extreme version of Inception

### Explainability Libraries
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **TensorFlow GradientTape**: For computing gradients
- **Matplotlib**: Visualization of heatmaps and results

### Data Processing
- **OpenCV (cv2)**: Image processing and manipulation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Seaborn**: Statistical data visualization

### Development Tools
- **Jupyter Notebooks**: Interactive development environment
- **Google Colab**: Cloud-based notebook execution
- **Matplotlib**: Plotting and visualization

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python** (3.7 or higher)
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify: `python --version`

2. **Required Python Packages**:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python scikit-learn
   ```

3. **Jupyter Notebook** (optional, for interactive development)
   ```bash
   pip install jupyter notebook
   ```

4. **Google Colab** (recommended for GPU access)
   - Access at [colab.research.google.com](https://colab.research.google.com/)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Afreen4115/ExplainableAI.git
cd ExplainableAI
```

### Step 2: Set Up Environment

**Option A: Local Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Google Colab (Recommended)**

1. Upload the notebook files to Google Drive
2. Open notebooks in Google Colab
3. Mount Google Drive to access data and notebooks

### Step 3: Download Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset:

1. **Kaggle Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. **Dataset Structure**:
   ```
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

3. **Update Path**: Modify the data directory path in notebooks:
   ```python
   main_dir = "/path/to/chest_xray/chest_xray"
   ```

### Step 4: Configure Environment Variables

For Google Colab, mount your drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 📖 Usage Guide

### Running the Main Model

1. **Open the Final Notebook**:
   - `BTP_B23APV04_Final.ipynb` - Complete ensemble model

2. **Execute Cells Sequentially**:
   - Data loading and preprocessing
   - Model architecture definition
   - Model compilation
   - Training
   - Evaluation

3. **Training Example**:
   ```python
   # Model will train for 20 epochs with early stopping
   stacked_history = stacked_model.fit(
       train_generator,
       steps_per_epoch=nb_train_samples // batch_size,
       epochs=20,
       validation_data=test_generator,
       callbacks=[EarlyStopping, model_save, rlr]
   )
   ```

### Running GradCAM Visualization

1. **Open GradCAM Notebook**:
   - `gradcam.ipynb` - Explainability visualization

2. **Load Trained Model**:
   ```python
   model = tf.keras.models.load_model('stacked_model.h5')
   ```

3. **Generate Heatmaps**:
   ```python
   heatmap = make_gradcam_heatmap(
       img_array, 
       model, 
       last_conv_layer_name
   )
   ```

4. **Visualize Results**:
   ```python
   plt.matshow(heatmap)
   plt.show()
   ```

### Model Architecture

#### Ensemble Model Structure

```
Input (224x224x3)
    ├── MobileNetV2 Base (frozen)
    │   └── GlobalAveragePooling2D
    │       └── Flatten → (1280,)
    │
    └── DenseNet169 Base (frozen)
        └── GlobalAveragePooling2D
            └── Flatten → (1664,)
                │
                └── Concatenate → (2944,)
                    ├── BatchNormalization
                    ├── Dense(256, ReLU)
                    ├── Dropout(0.5)
                    ├── BatchNormalization
                    ├── Dense(128, ReLU)
                    ├── Dropout(0.5)
                    └── Dense(1, Sigmoid) → Output
```

## 📁 Project Structure

```
ExplainableAI/
├── BTP_B23APV04_Final.ipynb    # Main ensemble model notebook
├── gradcam.ipynb                # GradCAM explainability notebook
├── model.ipynb                   # Base model implementation
├── model2.ipynb                  # Alternative model architecture
├── Reveiw.ipynb                  # Review and analysis notebook
├── try.ipynb                     # Experimental notebook
├── requirements.txt              # Python dependencies (to be created)
└── README.md                     # This file
```

## 🏗️ Architecture Details

### Ensemble Model Components

1. **MobileNetV2**:
   - Lightweight, mobile-optimized architecture
   - Pre-trained on ImageNet
   - Feature extraction layers frozen
   - Output: 1280-dimensional feature vector

2. **DenseNet169**:
   - Densely connected convolutional network
   - Pre-trained on ImageNet
   - Feature extraction layers frozen
   - Output: 1664-dimensional feature vector

3. **Fusion Layer**:
   - Concatenates features from both models
   - Creates 2944-dimensional combined feature vector

4. **Classification Head**:
   - Batch normalization layers
   - Dense layers with ReLU activation
   - Dropout for regularization (0.5)
   - Final sigmoid activation for binary classification

### Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Image Size**: 224x224 pixels
- **Epochs**: 20 (with early stopping)
- **Early Stopping**: Patience=6, monitor='val_accuracy'
- **Learning Rate Reduction**: Factor=0.01, patience=6

### Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values
    horizontal_flip=True      # Augment with horizontal flips
)
```

## 🔬 Explainability Methods

### GradCAM (Gradient-weighted Class Activation Mapping)

GradCAM generates visual explanations by:
1. Computing gradients of the predicted class score with respect to the last convolutional layer
2. Calculating the importance weights for each feature map
3. Creating a weighted combination of feature maps
4. Generating a heatmap showing important regions

**Implementation Steps**:
1. Load the trained model
2. Identify the last convolutional layer
3. Compute gradients using GradientTape
4. Generate heatmap from gradients
5. Overlay heatmap on original image

**Visualization**:
- Red regions: High importance for prediction
- Blue regions: Low importance
- Overlay on original X-ray for intuitive understanding

## 📊 Dataset Information

### Chest X-Ray Dataset

- **Total Images**: ~5,856 images
- **Training Set**: 5,216 images
  - Normal: ~1,341 images
  - Pneumonia: ~3,875 images
- **Validation Set**: 16 images
- **Test Set**: 624 images
  - Normal: ~234 images
  - Pneumonia: ~390 images

### Data Characteristics

- **Image Format**: JPEG
- **Image Dimensions**: Variable (resized to 224x224)
- **Color Channels**: RGB (3 channels)
- **Class Distribution**: Imbalanced (more pneumonia cases)

## 🧪 Model Performance

### Training Metrics

- **Training Accuracy**: ~96%+
- **Validation Accuracy**: ~85-87%
- **Training Loss**: Decreasing trend
- **Validation Loss**: Monitored for overfitting

### Model Evaluation

The ensemble model achieves:
- High accuracy on test set
- Good generalization (validation performance)
- Robust predictions with explainability

## 🔧 Configuration

### Modifying Model Architecture

Edit the model definition in the notebook:

```python
# Change base models
mobilenet_base = MobileNetV2(weights='imagenet', ...)
densenet_base = DenseNet169(weights='imagenet', ...)

# Modify dense layers
x = Dense(512, activation='relu')(x)  # Change units
x = Dropout(0.3)(x)  # Adjust dropout rate
```

### Adjusting Training Parameters

```python
# Change batch size
batch_size = 32

# Modify learning rate
optm = Adam(learning_rate=0.001)

# Adjust epochs
epochs = 30
```

### Customizing Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,        # Add rotation
    width_shift_range=0.2,    # Add width shift
    height_shift_range=0.2,   # Add height shift
    zoom_range=0.2            # Add zoom
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory Error
**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solutions**:
- Reduce batch size: `batch_size = 8`
- Use smaller image size: `img_width, img_height = [128, 128]`
- Use Google Colab with GPU runtime
- Enable mixed precision training

#### 2. Dataset Path Error
**Error**: `FileNotFoundError` or path issues

**Solutions**:
- Verify dataset path is correct
- Check folder structure matches expected format
- Use absolute paths instead of relative paths
- For Colab: Mount Google Drive correctly

#### 3. Model Loading Error
**Error**: `ValueError: Unknown layer` or model not found

**Solutions**:
- Ensure model file exists: `stacked_model.h5`
- Check model architecture matches saved model
- Re-train model if architecture changed
- Verify TensorFlow/Keras versions match

#### 4. GradCAM Not Working
**Error**: Layer name not found or gradients not computed

**Solutions**:
- Verify last convolutional layer name
- Check model architecture
- Ensure model is in inference mode
- Use correct layer name from model summary

#### 5. Slow Training
**Issue**: Training takes too long

**Solutions**:
- Use GPU runtime (Google Colab)
- Reduce image size
- Use smaller batch size
- Enable mixed precision
- Use fewer epochs with early stopping

#### 6. Overfitting
**Issue**: High training accuracy, low validation accuracy

**Solutions**:
- Increase dropout rate: `Dropout(0.7)`
- Add more data augmentation
- Use regularization techniques
- Reduce model complexity
- Increase validation data

## 🔒 Best Practices

### Model Development
- ✅ Use validation set for hyperparameter tuning
- ✅ Implement early stopping to prevent overfitting
- ✅ Save model checkpoints regularly
- ✅ Monitor training and validation metrics
- ✅ Use data augmentation for better generalization

### Explainability
- ✅ Always generate explanations for predictions
- ✅ Verify heatmaps align with medical knowledge
- ✅ Compare explanations across different models
- ✅ Document explainability findings

### Code Organization
- ✅ Use clear variable names
- ✅ Add comments for complex operations
- ✅ Organize code into logical cells
- ✅ Save intermediate results

## 🚧 Future Enhancements

Potential improvements for the project:

- [ ] Implement LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Add SHAP (SHapley Additive exPlanations) values
- [ ] Implement attention mechanisms
- [ ] Add more explainability methods (Integrated Gradients, etc.)
- [ ] Create web application for model deployment
- [ ] Add multi-class classification (bacterial vs. viral pneumonia)
- [ ] Implement model ensemble voting
- [ ] Add confidence scores to predictions
- [ ] Create interactive visualization dashboard
- [ ] Implement model versioning
- [ ] Add automated report generation
- [ ] Integrate with DICOM format support
- [ ] Add real-time inference API
- [ ] Implement federated learning
- [ ] Add uncertainty quantification
- [ ] Create mobile app for point-of-care diagnosis

## 📚 Research & References

### Key Papers
- **GradCAM**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **DenseNet**: "Densely Connected Convolutional Networks"
- **MobileNetV2**: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

### Datasets
- Chest X-Ray Images (Pneumonia) - Kaggle
- NIH Chest X-Ray Dataset

### Tools & Libraries
- TensorFlow/Keras Documentation
- GradCAM Implementation Guide
- Medical Image Analysis Resources



## 🙏 Acknowledgments

- **TensorFlow/Keras Team**: For the excellent deep learning framework
- **Dataset Contributors**: For providing the chest X-ray dataset
- **Research Community**: For explainability methods and techniques
- **Google Colab**: For providing free GPU resources
- Contributors and researchers in the explainable AI field



## 🔗 Useful Links

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)
- [Medical Image Analysis Resources](https://www.kaggle.com/datasets)
- [Google Colab](https://colab.research.google.com/)

## 🎯 Key Concepts Explained

### Transfer Learning
Using pre-trained models (trained on ImageNet) and fine-tuning them for medical image classification. This leverages learned features from natural images.

### Ensemble Learning
Combining predictions from multiple models (MobileNetV2 + DenseNet169) to improve accuracy and robustness.

### Explainable AI (XAI)
Making AI model decisions transparent and interpretable, crucial for medical applications where trust and understanding are essential.

### GradCAM
A technique that highlights important regions in images that influence model predictions, providing visual explanations.

### Data Augmentation
Techniques to artificially increase dataset size and diversity, improving model generalization.

---

**Making AI Transparent and Trustworthy in Healthcare! 🏥🤖✨**

