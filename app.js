// backend/app.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const apiRoutes = require('./routes/api'); // Make sure this path is correct
const imgRoutes = require('./routes/imgApi')
const { loadModels } = require('./utils/faceUtils');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// API Routes
app.use('/api', apiRoutes); // This now gets the properly exported router
app.use('/api', imgRoutes);
// console.log('ajay', apiRoutes)
// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Load face models and start server
loadModels()
  .then(() => {
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  })
  .catch(err => {
    console.error('Failed to load models:', err);
    process.exit(1);
  });