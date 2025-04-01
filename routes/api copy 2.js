// backend/routes/api.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const upload = multer();
const axios = require('axios');
const fs = require('fs');
const faceapi = require('face-api.js');
const { google } = require('googleapis');
const { Canvas, Image, createCanvas, loadImage } = require('canvas');
const canvas = require('canvas')
const tf = require('@tensorflow/tfjs-node');

// const { searchFaces } = require('../controllers/FaceController');

// POST endpoint for face search
router.post('/search', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }
        if (!req.body.folderId) {
            return res.status(400).json({ error: 'Missing Google Drive folder ID' });
        }
        
        const result = await searchFaces(req);
        res.json(result);
    } catch (error) {
        console.error('Error in /search route:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});


async function searchFaces(req) {
  try {
    // 1. Process uploaded image
    // const img = await loadImage(req.file.buffer);
    // const canvas = createCanvas(img.width, img.height);
    // const ctx = canvas.getContext('2d');
    // ctx.drawImage(img, 0, 0, img.width, img.height);

    const targetTensor = tf.node.decodeImage(req.file.buffer); // Convert Buffer to Tensor
    const targetDetections = await faceapi.detectAllFaces(targetTensor)
      .withFaceLandmarks()
      .withFaceDescriptors();
    targetTensor.dispose(); // Clean up memory


    if (targetDetections.length === 0) {
      return { matches: [], error: "No faces found in uploaded image" };
    }

    // 2. Configure Google Drive
    const auth = new google.auth.GoogleAuth({
      keyFile: './service-account.json',
      scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });
    const drive = google.drive({ version: 'v3', auth });

    // 3. Search Drive folder
    const { folderId } = req.body;
    const driveFiles = await drive.files.list({
      q: `'${folderId}' in parents and mimeType contains 'image/'`,
      fields: 'files(id, name, webContentLink)',
    });

    // 4. Compare faces with adjusted threshold (0.5)
    const matches = [];
    for (const file of driveFiles.data.files) {
      try {
        const response = await axios.get(file.webContentLink, {
          responseType: 'arraybuffer',
          timeout: 10000
        });

        // const driveImg = await loadImage(response.data);
        // const driveCanvas = createCanvas(driveImg.width, driveImg.height);
        // const driveCtx = driveCanvas.getContext('2d');
        // driveCtx.drawImage(driveImg, 0, 0, driveImg.width, driveImg.height);
        // console.log('response',response)
        const driveTensor = tf.node.decodeImage(response.data); // Convert Buffer to Tensor
        const driveDetections = await faceapi.detectAllFaces(driveTensor)
          .withFaceLandmarks()
          .withFaceDescriptors();
        driveTensor.dispose(); // Clean up memory


        if (driveDetections.length > 0) {
          const distance = faceapi.euclideanDistance(
            targetDetections[0].descriptor,
            driveDetections[0].descriptor
          );

        const getImg = (imgUrl) => {
          try {
            const fileId = imgUrl.split("id=")[1].split("&")[0]; // Extract file ID
            const directUrl = `https://drive.google.com/uc?export=view&id=${fileId}`;
            return directUrl
        } catch (error) {
            res.status(500).json({ error: "Failed to process image URL" });
        }
        }
          
          if (distance < 0.5) { // Adjusted threshold
            matches.push({
              fileId: file.id,
              fileName: file.name,
              similarity: Math.round((1 - distance) * 100),
              url: getImg(file.webContentLink)
            });
          }
        }
      } catch (err) {
        console.error(`Error processing ${file.name}:`, err.message);
      }
    }

    return { 
      matches: matches.sort((a, b) => b.similarity - a.similarity),
      debug: {
        detectedFaces: targetDetections.length,
        comparedFiles: driveFiles.data.files.length,
        successfulComparisons: matches.length
      }
    };

  } catch (error) {
    console.error('Search error:', error);
    return { error: "Face matching failed", details: error.message };
  }
}

// Export the router
module.exports = router;