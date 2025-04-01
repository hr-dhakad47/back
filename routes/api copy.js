// backend/routes/api.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const { searchFaces } = require('../controllers/FaceController');

const upload = multer();

// POST endpoint for face search
// In your backend route handler
router.post('/search', async (req, res) => {
    try {
      const { folderId } = req.body;
      const imageBuffer = req.file.buffer;
      
      // Load image and detect faces
      const img = await canvas.loadImage(imageBuffer);
      const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();
  
      // Validate detections
      if (!detections || detections.length === 0) {
        return res.status(400).json({
          error: 'No faces detected in the uploaded image',
          debug: ['Face detection completed - no faces found']
        });
      }
  
      // Process each detection
      const validDetections = detections.filter(det => 
        det.detection && det.detection.box && 
        typeof det.detection.box.x === 'number'
      );
  
      if (validDetections.length === 0) {
        return res.status(400).json({
          error: 'Invalid face detection results',
          debug: ['Faces detected but bounding boxes were invalid']
        });
      }
  
      // Rest of your matching logic...
      
    } catch (error) {
      console.error('Search error:', error);
      res.status(500).json({ error: error.message });
    }
  });
// Export the router
module.exports = router;