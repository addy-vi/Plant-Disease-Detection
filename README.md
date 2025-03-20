# Plant Disease Detection

## Introduction
Plant diseases have a significant impact on agriculture, leading to reduced crop yield and economic losses. Early and accurate detection of plant diseases can help farmers take necessary actions to mitigate damage. This project focuses on using technology, such as machine learning and image processing, to identify plant diseases from leaf images.



## Objectives
- Develop a system to detect plant diseases from leaf images.
- Utilize machine learning models for classification.
- Provide actionable insights to farmers for disease prevention and control.



## Technologies Used
- **Programming Language:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Tools:** Jupyter Notebook, Streamlit
- **Dataset:** PlantVillage Dataset or any relevant image dataset of diseased and healthy leaves.



## Workflow
1. **Data Collection:**
   - Gather images of healthy and diseased plant leaves.
   - Use publicly available datasets or collect your own data.

2. **Data Preprocessing:**
   - Resize images for consistency.
   - Normalize image pixel values.
   - Perform data augmentation.

3. **Model Development:**
   - Choose a suitable machine learning model that is CNN.
   - Train the model using the preprocessed dataset.
   - Evaluate model performance using metrics such as accuracy and F1-score.

4. **Disease Detection:**
   - Use the trained model to classify new images.
   - Implement a user-friendly interface for farmers to upload images.

5. **Result Interpretation:**
   - Display disease type and provide actionable recommendations.
   - Optionally integrate with IoT for real-time monitoring.



## Results
- Accuracy achieved on test dataset: **96%**



## Future Work
- Expand the dataset to include more plant species and diseases.
- Improve model accuracy using advanced techniques like transfer learning.
- Deploy the system as a mobile app for farmers' convenience.



## Conclusion
This project aims to bridge the gap between technology and agriculture by providing an efficient solution for plant disease detection. By leveraging machine learning, we can help farmers take timely action, ultimately contributing to food security and sustainable agriculture.



## References
1. PlantVillage Dataset: [Link](https://www.plantvillage.org/)
2. TensorFlow Documentation: [Link](https://www.tensorflow.org/)
3. OpenCV Library: [Link](https://opencv.org/)



## Clone Repository

To clone this repository to your local machine, run the following command:

```bash
git clone https://github.com/addy-vi/Plant-Disease-Detection.git
