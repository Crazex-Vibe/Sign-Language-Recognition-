# Sign Language Recognition

This project is a real-time sign language recognition system that detects hand gestures and translates them into text. It also features a voice AI that reads out detected sentences.

## Features
- Real-time hand tracking using **MediaPipe**
- Integration with **Sign Language MNIST Dataset**
- Machine learning model for gesture classification
- **Tkinter UI** for visual feedback and text display
- **Voice AI** for reading out translated sentences

## Project Structure
- `Dataset_create.py` - Generates dataset for training
- `Model_Train.py` - Trains the model for sign recognition
- `Pickle_Data.py` - Processes and saves dataset in a pickle format
- `TestVer.py` - Main GUI application for real-time sign language translation
- `model.p` - Trained model file
- `data.pickle` - Preprocessed dataset file

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Crazex-Vibe/Sign-Language-Recognition.git
   cd Sign-Language-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python TestVer.py
   ```

## Usage
1. Start the application.
2. Show hand gestures corresponding to sign language letters.
3. The detected letter will appear on the UI.
4. Press the **'Speak'** button to hear the recognized sentence aloud.

## Future Enhancements
- Improve recognition accuracy
- Support more sign languages
- Implement a more advanced UI

## Contributing
Feel free to contribute by creating a pull request or submitting issues.

## License
This project is licensed under the MIT License.

---
Created by **Crazex-Vibe**

