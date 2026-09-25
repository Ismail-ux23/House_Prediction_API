const form = document.querySelector('#prediction-form');
const state = document.querySelector('#result-state');
const price = document.querySelector('#predicted-price');
const shortPrice = document.querySelector('#predicted-short');
const range = document.querySelector('#range-value');
const marker = document.querySelector('#range-marker');
const errorMessage = document.querySelector('#error-message');

if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitButton = form.querySelector('button[type="submit"]');
    const formData = new FormData(form);
    const payload = Object.fromEntries(
      [...formData.entries()].map(([key, value]) => [key, Number(value)])
    );

    submitButton.disabled = true;
    submitButton.querySelector('span:first-child').textContent = 'Reading signals...';
    state.textContent = 'Calculating';
    errorMessage.textContent = '';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'The estimate could not be calculated.');

      price.textContent = result.predicted_price;
      shortPrice.textContent = result.predicted_price_short;
      range.textContent = result.fidence_range;
      marker.classList.add('is-visible');
      state.textContent = 'Updated now';
    } catch (error) {
      state.textContent = 'Needs attention';
      errorMessage.textContent = error.message;
    } finally {
      submitButton.disabled = false;
      submitButton.querySelector('span:first-child').textContent = 'Run estimate';
    }
  });
}

const uploadForm = document.querySelector('#upload-form');
const csvFile = document.querySelector('#csv-file');
const fileName = document.querySelector('#file-name');
const fileDrop = document.querySelector('#file-drop');
const fileMeta = document.querySelector('#file-meta');
const uploadMessage = document.querySelector('#upload-message');
const uploadMessageText = document.querySelector('#upload-message-text');
const batchResult = document.querySelector('#batch-result');
const batchCount = document.querySelector('#batch-count');
const batchTable = document.querySelector('#batch-table');
const downloadLink = document.querySelector('#download-link');
let predictionUrl = null;

const setUploadMessage = (message, status = '') => {
  uploadMessage.className = `upload-message ${status}`;
  uploadMessageText.textContent = message;
};

const showSelectedFile = (file) => {
  if (!file) {
    fileName.textContent = 'Drop a CSV here or choose a file';
    fileMeta.textContent = 'Required columns are listed in the Guide';
    return;
  }
  fileName.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB CSV ready to process`;
  setUploadMessage('Ready to predict this file', 'ready');
};

if (csvFile) {
  csvFile.addEventListener('change', () => showSelectedFile(csvFile.files[0]));
}

if (fileDrop) {
  ['dragenter', 'dragover'].forEach((eventName) => fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDrop.classList.add('is-dragging');
  }));
  ['dragleave', 'drop'].forEach((eventName) => fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDrop.classList.remove('is-dragging');
  }));
  fileDrop.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
      setUploadMessage('Please choose a .csv file', 'error');
      return;
    }
    const transfer = new DataTransfer();
    transfer.items.add(file);
    csvFile.files = transfer.files;
    showSelectedFile(file);
  });
}

if (uploadForm) {
  uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitButton = uploadForm.querySelector('button');
    const selectedFile = csvFile.files[0];
    if (!selectedFile) {
      setUploadMessage('Choose a CSV file before predicting', 'error');
      return;
    }

    const body = new FormData();
    body.append('file', selectedFile);
    submitButton.disabled = true;
    submitButton.querySelector('span:first-child').textContent = 'Predicting...';
    setUploadMessage('Processing rows through the model...', 'loading');

    try {
      const response = await fetch('/predict_file', { method: 'POST', body });
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || 'The CSV could not be processed.');
      }
      const resultBlob = await response.blob();
      const csvText = await resultBlob.text();
      const rows = parseCsv(csvText);
      const headers = rows[0] || [];
      const dataRows = rows.slice(1);
      batchCount.textContent = `${dataRows.length} row${dataRows.length === 1 ? '' : 's'} predicted`;
      batchTable.innerHTML = `<thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead><tbody>${dataRows.slice(0, 4).map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('')}</tbody>`;
      batchResult.hidden = false;
      if (predictionUrl) URL.revokeObjectURL(predictionUrl);
      predictionUrl = URL.createObjectURL(resultBlob);
      downloadLink.href = predictionUrl;
      const download = document.createElement('a');
      download.href = predictionUrl;
      download.download = 'predictions.csv';
      document.body.appendChild(download);
      download.click();
      download.remove();
      setUploadMessage('Done. predictions.csv was downloaded.', 'success');
    } catch (error) {
      setUploadMessage(error.message, 'error');
    } finally {
      submitButton.disabled = false;
      submitButton.querySelector('span:first-child').textContent = 'Predict CSV';
    }
  });
}

function parseCsv(text) {
  return text.trim().split(/\r?\n/).map((line) => {
    const values = [];
    let value = '';
    let quoted = false;
    for (const character of line) {
      if (character === '"') quoted = !quoted;
      else if (character === ',' && !quoted) { values.push(value); value = ''; }
      else value += character;
    }
    values.push(value);
    return values;
  });
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[character]));
}
