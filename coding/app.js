// node --version # Should be >= 18
// npm install @google/generative-ai express dotenv fs path pdf2json

const express = require('express');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const PDFParser = require('pdf2json'); 
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const API_KEY = process.env.API_KEY;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const genAI = new GoogleGenerativeAI(API_KEY);
const MODEL_NAME = "gemini-1.5-flash";

// Function to extract text from a PDF file
const extractTextFromPdf = (filePath) => {
  return new Promise((resolve, reject) => {
    const pdfParser = new PDFParser();
    pdfParser.on('pdfParser_dataError', errData => reject(errData.parserError));
    pdfParser.on('pdfParser_dataReady', pdfData => {
      const rawText = pdfParser.getRawTextContent();
      resolve(rawText);
    });
    pdfParser.loadPDF(filePath);
  });
};

// Function to preload file contents
const preloadFileContents = async (filePaths) => {
  const preloadedTexts = {};
  for (const filePath of filePaths) {
    const fullPath = path.resolve(__dirname, filePath);
    if (!fs.existsSync(fullPath)) {
      console.error(`File not found: ${fullPath}`);
      continue;
    }
    const ext = path.extname(filePath);
    let fileData;
    if (ext === '.pdf') {
      fileData = await extractTextFromPdf(fullPath);
    } else {
      fileData = fs.readFileSync(fullPath, 'utf-8');
    }
    preloadedTexts[filePath] = fileData;
  }
  return preloadedTexts;
};

// Paths to preloaded files
const preloadedFilePaths = ['sample.txt'];
let preloadedTexts = {};

// Preload file contents
preloadFileContents(preloadedFilePaths).then((texts) => {
  preloadedTexts = texts;
});

// Function to filter text
const filterText = (text) => {
  const harmfulKeywords = ['sex', 'hate', 'harass', 'dangerous'];
  return text.split(' ').filter(word => !harmfulKeywords.includes(word.toLowerCase())).join(' ');
};

// Combine preloaded and filtered texts into a single context
const filteredPreloadedText = () => Object.values(preloadedTexts).map(filterText).join(' ');

// Function to generate a response from the context using Gemini API
const generateResponseFromContext = async (query, context) => {
  try {
    const model = genAI.getGenerativeModel({ model: MODEL_NAME, max_tokens: 150,
      temperature: 0.7});
    const response = await model.generateContent({
      prompt: `${context}\n\nUser: ${query}\nAI:`,
    });
    return response.data.choices[0].text.trim();
  } catch (error) {
    console.error('Error generating response:', error);
    return 'Error: Failed to generate response. Please try again later.';
  }
};

// Function to handle user queries
const handleUserQuery = async (query) => {
  return await generateResponseFromContext(query, filteredPreloadedText());
};

// Route to serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

// Route to serve loader GIF
app.get('/loader.gif', (req, res) => {
  res.sendFile(__dirname + '/loader.gif');
});

// Route to handle chat functionality
app.post('/chat', async (req, res) => {
  try {
    const userInput = req.body?.userInput;
    console.log('Incoming /chat request:', userInput);
    if (!userInput) {
      return res.status(400).json({ error: 'Invalid request body' });
    }

    const chatResponse = await handleUserQuery(userInput);
    res.json({ response: chatResponse });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Route to handle user queries
app.post('/query', async (req, res) => {
  const { query } = req.body;
  try {
    const response = await handleUserQuery(query);
    res.json({ response });
  } catch (error) {
    console.error('Error handling query:', error);
    res.status(500).json({ error: 'Failed to generate response. Please try again later.' });
  }
});

// Route to handle summarization
app.post('/summarize', async (req, res) => {
  const { input_text } = req.body;
  try {
    const summary = await generateResponseFromContext(input_text, filteredPreloadedText());
    res.json({ summary });
  } catch (error) {
    console.error('Error summarizing text:', error);
    res.status(500).json({ error: 'Failed to summarize text. Please try again later.' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
