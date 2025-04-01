const express = require('express');
const router = express.Router();
const multer = require('multer');
const upload = multer();
const fs = require('fs');
const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const IMAGE_FOLDER = path.join(__dirname, '..', 'images'); // Local image storage
const imageFiles = fs.readdirSync(IMAGE_FOLDER);

// Helper function to load an image from buffer
async function loadImage(buffer) {
    const img = await canvas.loadImage(buffer);
    return faceapi.createCanvasFromMedia(img);
}

// Load face-api models
async function loadModels() {
    const modelPath = path.join(__dirname, '..', 'models'); // Ensure models are stored in this directory
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}

// Load models before handling requests
loadModels().then(() => console.log("FaceAPI models loaded."));

// POST endpoint for face search
router.post('/search', upload.single('image'), async (req, res) => {
    try {
        // Step 1: Extract all faces from the uploaded image
        const targetCanvas = await loadImage(req.file.buffer);
        const targetDetections = await faceapi.detectAllFaces(targetCanvas)
            .withFaceLandmarks()
            .withFaceDescriptors();

        if (targetDetections.length === 0) {
            return res.json({ matches: [], error: "No faces found in uploaded image" });
        }

        const matches = [];

        // Step 2: Loop through each stored image to find a match
        for (const fileName of imageFiles) {
            try {
                const filePath = path.join(IMAGE_FOLDER, fileName);
                const imageBuffer = fs.readFileSync(filePath);

                const driveCanvas = await loadImage(imageBuffer);
                const driveDetections = await faceapi.detectAllFaces(driveCanvas)
                    .withFaceLandmarks()
                    .withFaceDescriptors();

                // Step 3: Compare EACH face in the uploaded image against EACH face in the stored image
                for (const targetFace of targetDetections) {
                    for (const driveFace of driveDetections) {
                        const distance = faceapi.euclideanDistance(
                            targetFace.descriptor,
                            driveFace.descriptor
                        );

                        if (distance < 0.5) { // Match threshold
                            matches.push({
                                fileName,
                                similarity: Math.round((1 - distance) * 100),
                                fileBuffer: imageBuffer.toString('base64') // Convert buffer to Base64
                            });

                            // Stop checking once we find a match in the image
                            break;
                        }
                    }
                }
            } catch (err) {
                console.error(`Error processing ${fileName}:`, err.message);
            }
        }

        res.json({
            matches: matches.sort((a, b) => b.similarity - a.similarity),
            debug: {
                detectedFaces: targetDetections.length,
                comparedFiles: imageFiles.length,
                successfulComparisons: matches.length
            }
        });
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ error: "Face matching failed", details: error.message });
    }
});

module.exports = router;
