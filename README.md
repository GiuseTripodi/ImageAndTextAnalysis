<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Analysis of Images and Text</h3>
    Image and text analysis performed using various classification and anomaly detection techniques.
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Structure">Structure</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The project carried out for the course "Machine and Deep Learning," consists of analyzing two types of datasets: images and text, using different machine learning algorithms.

### Structure
- **models_images**: Folder with pre-trained models for images.
- **models_text**: Folder with pre-trained models for texts.
- **notebook**: Folder with notebooks used in the development of the project.
- *accuracy_images_script.py*: script to compute accuracy on custom images dataset with the pretrained models.
- *accuracy_text_script.py*: script to compute accuracy on custom texts dataset with the pretrained models



### Built With

* [scikit learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/tes)
* [TensorFlow](https://www.tensorflow.org/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Install the following libraries.

* requirements.txt
  ```sh
    pip install -r requirements.txt
  ```

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/GiuseTripodi/Image_Text_Analysis.git
   ```



<!-- USAGE EXAMPLES -->
## Usage

Run one of the two scripts:
- *accuracy_images_script.py*: If you want to calculate accuracy on images with pretrained models.
- *accuracy_text_script.py*: If you want to calculate accuracy on text with pretrained models.

Information about script.
  ```sh
    usage: accuracy_images_script.py [-h] [-s SCALE] [--version] images_path models_path

    positional arguments:
      images_path           path of the directory with all the images
      models_path           path of the directory with all the images

    optional arguments:
      -h, --help            show this help message and exit
      -s SCALE, --scale SCALE
                        Percentage of the data to get, is a value between (0,1)
      --version             show program's version number and exit
  ```

To run the script:
  ```sh
    python <script> [-s] <images_path> <models_path>
  ```

Example:
  ```sh
     python accuracy_images_script.py --scale 0.05  "/home/giuseppe/Scrivania/universita/Magistrale/Machine and Deep Learning/Progetto ML/Dataset/immagini-3/immagini-3" "/home/giuseppe/Scrivania/model_accuracy/models_images"

  ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Giuseppe Tripodi - [@giuseppetripod3](https://twitter.com/giuseppetripod3) - giuseppetripodi1@outlook.it - [LinkedIn](https://www.linkedin.com/in/giuseppe-tripodi-unical/)




