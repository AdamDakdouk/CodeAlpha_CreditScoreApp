import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'Server is running!', timestamp: new Date() });
});

// Credit prediction endpoint
app.post('/api/predict', async (req, res) => {
  try {
    const {
      age,
      income,
      debt_to_income,
      credit_history_length,
      num_credit_accounts,
      payment_history_score,
      credit_utilization,
      num_late_payments,
      employment_years
    } = req.body;

    // Validate required fields
    const requiredFields = [
      'age', 'income', 'debt_to_income', 'credit_history_length',
      'num_credit_accounts', 'payment_history_score', 'credit_utilization',
      'num_late_payments', 'employment_years'
    ];

    for (const field of requiredFields) {
      if (req.body[field] === undefined || req.body[field] === null) {
        return res.status(400).json({ 
          error: `Missing required field: ${field}` 
        });
      }
    }

    // Path to Python script and model directory
    const mlModelDir = path.join(__dirname, '../../ml-model');
    const pythonScriptPath = path.join(mlModelDir, 'credit_model.py');
    
    // Prepare data for Python script
    const userData = JSON.stringify({
      age: Number(age),
      income: Number(income),
      debt_to_income: Number(debt_to_income),
      credit_history_length: Number(credit_history_length),
      num_credit_accounts: Number(num_credit_accounts),
      payment_history_score: Number(payment_history_score),
      credit_utilization: Number(credit_utilization),
      num_late_payments: Number(num_late_payments),
      employment_years: Number(employment_years)
    });

    // Call Python script with correct working directory
    const python = spawn('python', [pythonScriptPath, userData], {
      cwd: mlModelDir  // This sets the working directory to ml-model folder
    });
    
    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    python.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script error:', errorString);
        return res.status(500).json({ 
          error: 'Prediction service unavailable',
          details: errorString 
        });
      }

      try {
        // Parse Python script output
        const result = JSON.parse(dataString.trim());
        
        res.json({
          success: true,
          prediction: result,
          timestamp: new Date()
        });
      } catch (parseError) {
        console.error('Failed to parse Python output:', dataString);
        res.status(500).json({ 
          error: 'Invalid prediction response',
          details: dataString 
        });
      }
    });

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 ML Model API ready at http://localhost:${PORT}/api/predict`);
});