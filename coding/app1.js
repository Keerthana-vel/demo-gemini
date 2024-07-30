const express = require('express');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const PDFParser = require('pdf2json'); 
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold  } = require("@google/generative-ai");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const genAI = new GoogleGenerativeAI(process.env.API_KEY);

const generationconfig = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
const MODEL_NAME = "gemini-1.5-flash"; // Ensure the correct model is specified

// Function to extract text from a PDF file
async function extractTextFromPdf(filePath) {
  return new Promise((resolve, reject) => {
    const pdfParser = new PDFParser();
    pdfParser.on('pdfParser_dataError', errData => reject(errData.parserError));
    pdfParser.on('pdfParser_dataReady', pdfData => {
      const rawText = pdfParser.getRawTextContent();
      resolve(rawText);
      return pdfData;
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
    console.log("111111111",fileData)
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
console.log('jijjiji',preloadedTexts);
});


// Function to filter text
const filterText = (text) => {
  const harmfulKeywords = ['sex', 'hate', 'harass', 'dangerous'];
  return text.split(' ').filter(word => !harmfulKeywords.includes(word.toLowerCase())).join(' ');
};

// Combine preloaded and filtered texts into a single context
const filteredPreloadedText = () => Object.values(preloadedTexts).map(filterText).join(' ');

// Function to handle user queries
const handleUserQuery = async (query) => {
  return await generateResponseFromContext(query, filteredPreloadedText());
};

// Function to generate a response from the context using Gemini API
const generateResponseFromContext = async (query, context) => {
  try {
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });
    // Improved prompt 
  const server = `Please respond to the following user query, using only the information provided in the context. Avoid generating content that could be considered harmful or unsafe.
  Context: ${context}\n\n
  User: ${query}\n
  Response:`;
  console.log('2222222',context);
  // Generate content
  const response = await model.generateContent(server, {}, generationconfig);
    console.log('Full Response:4444444444', JSON.stringify(response, null, 2));

    // Access the nested properties
    if (response && response.response && response.response.candidates && response.response.candidates.length > 0) {
      const generatedContent = response.response.candidates[0].content || response.response.candidates[0].text || '';
      console.log('Candidate Object:', JSON.stringify(response.response.candidates, null, 2));
      console.log('Generated Content:', response);

      return generatedContent;
    } else {
      console.log('No candidates found in the response.');
      return 'No response generated.';
    }

  } catch (error) {
    console.error('Error generating response:', error);
  }
};

    

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
// Route to serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

// Route to serve loader GIF
app.get('/loader.gif', (req, res) => {
  res.sendFile(__dirname + '/loader.gif');
});
// Express endpoint to handle user queries
app.post('/chat', async (req, res) => {
  try {
    const userInput = req.body?.userInput;
    // console.log('Incoming /chat request:', userInput);
    if (!userInput) {
      return res.status(400).json({ error: 'Invalid request body' });
    }

    const response = await handleUserQuery(userInput);
    res.json({ response });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
