const statusEl = document.getElementById('status');
const apiKeyInput = document.getElementById('apiKey');
const inputText = document.getElementById('inputText');

async function scanText() {
  const apiKey = apiKeyInput.value.trim();
  const text = inputText.value.trim();
  if (!apiKey || !text) {
    statusEl.textContent = 'API key and text are required.';
    return;
  }

  statusEl.textContent = 'Scanning...';
  try {
    const response = await fetch('http://localhost:5001/api/scan/text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': apiKey
      },
      body: JSON.stringify({ text })
    });
    const data = await response.json();
    statusEl.textContent = data.result ? `${data.result} (${data.confidence}%)` : 'Scan failed';
  } catch (error) {
    statusEl.textContent = 'Connection error.';
  }
}

async function scanUrl() {
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    statusEl.textContent = 'API key required.';
    return;
  }

  statusEl.textContent = 'Fetching current URL...';
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0]?.url;
    if (!url) {
      statusEl.textContent = 'Unable to read current URL.';
      return;
    }
    statusEl.textContent = 'Scanning URL...';
    try {
      const response = await fetch('http://localhost:5001/api/scan/url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': apiKey
        },
        body: JSON.stringify({ url })
      });
      const data = await response.json();
      statusEl.textContent = data.result ? `${data.result} (${data.confidence}%)` : 'Scan failed';
    } catch (error) {
      statusEl.textContent = 'Connection error.';
    }
  });
}

document.getElementById('scanText').addEventListener('click', scanText);
document.getElementById('scanUrl').addEventListener('click', scanUrl);
