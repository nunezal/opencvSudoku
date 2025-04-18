# OpenCV Sudoku Solver

A computer vision application that solves Sudoku puzzles in real-time from a video feed. The system captures Sudoku puzzles through a webcam, processes the image to identify the grid and digits, solves the puzzle, and overlays the solution onto the original video feed.

## Features

- Real-time Sudoku puzzle detection from webcam feed
- Digit recognition using a trained deep neural network
- Automatic grid extraction and perspective correction
- Fast puzzle solving algorithm based on constraint propagation
- Augmented reality display of the solution overlaid on the original puzzle

## Architecture

The system consists of three main components:

1. **Image Processing (OpenCV)**: Detects the Sudoku grid in the video frame, extracts it, and prepares the individual cells for digit recognition.
2. **Digit Recognition (TensorFlow)**: Identifies the digits present in the Sudoku grid using a trained neural network.
3. **Sudoku Solver**: Implements a constraint propagation and search algorithm to solve the puzzle efficiently.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Keras/TensorFlow
- Webcam or video input device

## Usage

Run the main application:

```
python SodukuFinder.py
```

Hold a Sudoku puzzle up to your webcam. The application will:

1. Detect the Sudoku grid
2. Extract and recognize the digits
3. Solve the puzzle
4. Display the solution overlaid on the original image

## How It Works

### Grid Detection

The application uses contour detection to identify the largest square-like shape in the frame, which is assumed to be the Sudoku grid.

### Digit Recognition

A convolutional neural network (CNN) trained on the MNIST dataset and customized for Sudoku digit recognition is used to identify the digits within each cell of the grid.

### Solving Algorithm

The solver uses Peter Norvig's algorithm which combines constraint propagation with a depth-first search strategy for efficient solving:

1. Constraint propagation eliminates invalid digit placements
2. Search explores possible solutions when constraints alone aren't sufficient
3. The algorithm is optimized to solve even the most difficult puzzles quickly

## Project Structure

- `SodukuFinder.py`: Main application with image processing and UI
- `Soduku_solver.py`: Implementation of the Sudoku solving algorithm
- `model.json` & `model.h5`: Neural network model for digit recognition

## Acknowledgments

- Peter Norvig's Sudoku solver algorithm: http://norvig.com/sudoku.html
- TensorFlow and Keras for the neural network framework
- OpenCV library for computer vision capabilities
