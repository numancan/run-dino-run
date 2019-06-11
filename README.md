# AI learn how to play Google's dinosaur game
A simple artificial intelligence to teach Google Chrome's dinosaur game to play, using [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) (Neuroevolution of augmenting topologies) algorithm.

<p align="center"> 
<img src="images/winner.gif">
</p>

## How it works

<p align="center"> 
<img src="images/how-it-works.png">
</p>


## Dependencies
This project requires the following dependencies:
* [Python](https://www.python.org/downloads)(>= 3)
* [NumPy](http://www.numpy.org)
* [OpenCV](https://opencv.org/releases/)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* [pyautogui](https://pyautogui.readthedocs.io/en/latest/install.html)
* [neat-python](https://neat-python.readthedocs.io/en/latest/installation.html)
* [graphviz](https://pypi.org/project/graphviz/)
* [Graphviz executable](https://www.graphviz.org/download/)(==2.38)

*You have to install both graphviz library and graphviz's executable package*

## How To Use

* ### Install and prepare game
  1- Go to [t-rex-runner](https://github.com/wayou/t-rex-runner) and download zip file.

  2- Extract zip file.

  3- Open **index.js** with notepad or any text editor.

  4- Change all code in the index.js with [this](https://raw.githubusercontent.com/numancan/run-dino-run/master/__pycache__/index.txt?token=AJBXZ6GFU65WKDMVSQTO6WK47SPHW).

* ### Set position of game window
  1- Open **index.html**. You have to shrink the page like this:
  
  ![shrink](images/shrink.gif)
  
  2- Set position
  
  ![shrink](images/set.gif)

* ### Train

  1-Run **train.py**. 
  
* ### Run trained

```
python runwinner.py best-genomes/best_378_2544.71.pkl
```

