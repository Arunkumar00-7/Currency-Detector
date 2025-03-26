import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera, Volume2, VolumeX } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite';

function App() {
  const webcamRef = useRef<Webcam>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [lastDetection, setLastDetection] = useState<string>('');
  const [model, setModel] = useState<tflite.TFLiteModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [showAbout, setShowAbout] = useState(false); // State for About content visibility

  // Load available voices for speech synthesis
  useEffect(() => {
    const loadVoices = () => {
      const availableVoices = window.speechSynthesis.getVoices();
      setVoices(availableVoices);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  // Load the TFLite model
  useEffect(() => {
    const loadModel = async (): Promise<void> => {
  try {
    setIsLoading(true);
    await tf.ready();

    const modelPath = './Currency-Detector/model.tflite';
    console.log('Loading model from:', modelPath);
    const loadedModel = await tflite.loadTFLiteModel(modelPath);

    setModel(loadedModel);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    setError('Failed to load detection model. Please check the model path.');
  } finally {
    setIsLoading(false);
  }
};

    loadModel();
  }, []);

  // Function to provide voice feedback
  const speak = (text: string): void => {
    if (isMuted || !text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.voice = voices.find(v => v.lang.includes('en-IN')) || voices[0];
    window.speechSynthesis.speak(utterance);
  };

  // Preprocess the webcam image for the model
  const preprocessImage = (video: HTMLVideoElement): tf.Tensor => {
    return tf.tidy(() => {
      return tf.browser.fromPixels(video)
        .resizeBilinear([224, 224]) // Resize to match model input
        .toFloat()
        .div(255) // Normalize pixel values
        .expandDims(0); // Add batch dimension
    });
  };

  // Perform currency detection
  const detectCurrency = async (restart: boolean = false): Promise<void> => {
    if (!webcamRef.current || !model || !webcamRef.current.video) return;
    if (!webcamRef.current || !model || !webcamRef.current.video) return;
    try {
      const tensor = preprocessImage(webcamRef.current.video);
      const predictions = model.predict(tensor) as tf.Tensor | tf.Tensor[] | NamedTensorMap;
      const scores = await (predictions as tf.Tensor).data();

      // Mapping output index to currency type
      const currencyMap = {
        0: '100 Rupee note',
        1: '200 Rupee note',
        2: '500 Rupee note'
      };

      const topIndex = scores.indexOf(Math.max(...scores));
      const detectedCurrency = currencyMap[topIndex as keyof typeof currencyMap];

      if (detectedCurrency && detectedCurrency !== lastDetection) {
        setIsDetecting(false); // Stop detecting after a successful detection
        setLastDetection(detectedCurrency);
        speak(`Detected ${detectedCurrency}`);
      }
      
      tf.dispose([predictions, tensor]);
    } catch (error) {
      console.error('Error during detection:', error);
      setError('Detection error occurred. Please try again.');
    }
  };

  // Run detection every second when detecting is enabled
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isDetecting && !isLoading && !error) {
      interval = setInterval(detectCurrency, 1000);
    }
    return () => clearInterval(interval);
  }, [isDetecting, isLoading, error, model]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto p-8">
        <h1 className="text-4xl font-bold text-center mb-6">Indian Currency Detector</h1>
        
        <button
          onClick={() => setShowAbout(!showAbout)} // Toggle About content
          className="mb-4 p-2 bg-blue-500 rounded"
        >
          About
        </button>

        {showAbout && (
          <div className="bg-gray-800 p-4 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-2">About This Project</h2>
            <p>
              This project is designed to detect Indian currency notes using a webcam. 
              It utilizes TensorFlow.js for machine learning and provides voice feedback 
              for detected currency. The application is built with React and is intended 
              to assist users in identifying currency notes quickly and efficiently.
            </p>
          </div>
        )}

        {isLoading ? (
          <p>Loading model...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : (
          <div className="relative rounded-lg overflow-hidden shadow-xl bg-gray-800 mb-6">
            {/* Webcam Feed */}
            <Webcam
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="w-full h-auto"
              videoConstraints={{
                facingMode: 'environment' // Use back camera if available
              }}
            />
            
            {/* Buttons for starting detection and muting speech */}
            <div className="absolute bottom-4 left-4 right-4 flex justify-between">
              <button
                onClick={() => {
                  setIsDetecting(!isDetecting);
                  if (!isDetecting) detectCurrency(true); // Start detection
                }}
                className="p-3 rounded-full bg-green-500"
              >
                <Camera className="w-6 h-6" />
              </button>
              
              <button
                onClick={() => setIsMuted(!isMuted)}
                className="p-3 rounded-full bg-blue-500"
              >
                {isMuted ? <VolumeX className="w-6 h-6" /> : <Volume2 className="w-6 h-6" />}
              </button>
            </div>
          </div>
        )}
        
        {/* Detection Status */}
        <p className="text-xl text-center">
          {lastDetection || 'No Indian currency detected yet'}
        </p>
      </div>
    </div>
  );
}

export default App;
