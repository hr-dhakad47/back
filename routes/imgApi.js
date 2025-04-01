// backend/routes/api.js
const express = require('express');
const router = express.Router();
const axios = require("axios");
// const cors = require("cors");

// console.log('apiroutes')
router.get('/getImageUrl', async (req, res) => {
    
    const { imgUrl } = req.query;
    
    if (!imgUrl) {
        return res.status(400).json({ error: "Missing imgUrl parameter" });
    }
    
    try {
        // Follow the redirect to get the final image URL
        const response = await axios.head(imgUrl, { maxRedirects: 5 });
        const finalUrl = response.request.res.responseUrl;
        
        res.json({ finalUrl });
    } catch (error) {
        console.error("Error fetching image URL:", error.message);
        res.status(500).json({ error: "Failed to fetch image URL" });
    }
});


module.exports = router;