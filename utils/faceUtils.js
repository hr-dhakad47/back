// backend/utils/faceUtils.js
const faceapi = require('face-api.js');
const path = require('path');

async function loadModels() {
  try {
    const modelsPath = path.join(__dirname, '../models');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);
    console.log('Face models loaded successfully');
  } catch (err) {
    console.error('Failed to load models:', err);
    throw err;
  }
}

module.exports = { loadModels };