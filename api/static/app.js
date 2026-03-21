const dropzone = document.getElementById('dropzone');
const input = document.getElementById('video-input');
const filenameEl = document.getElementById('filename');
const submitBtn = document.getElementById('submit-btn');
const statusEl = document.getElementById('status');
const resultCard = document.getElementById('result-card');
const resultLink = document.getElementById('result-link');
const resultVideo = document.getElementById('result-video');

let selectedFile = null;
const maxSizeMb = Number(dropzone.dataset.maxSize || '150');
const maxSizeBytes = maxSizeMb * 1024 * 1024;

const setStatus = (msg, type = '') => {
  statusEl.textContent = msg;
  statusEl.className = `status ${type}`;
};

const setFile = (file) => {
  if (!file) return;
  if (file.size > maxSizeBytes) {
    setStatus(`File is too large (>${maxSizeMb} MB).`, 'error');
    selectedFile = null;
    filenameEl.textContent = 'No file selected.';
    return;
  }
  selectedFile = file;
  filenameEl.textContent = `${file.name} — ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  setStatus('Ready to upload.');
};

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.querySelector('.droparea').classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.querySelector('.droparea').classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  setFile(file);
});

dropzone.addEventListener('click', () => input.click());
input.addEventListener('change', (e) => setFile(e.target.files[0]));

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    setStatus('Pick a .mp4, .mov, or .mkv file first.', 'error');
    return;
  }

  const form = new FormData();
  form.append('video', selectedFile, selectedFile.name);

  submitBtn.disabled = true;
  setStatus('Uploading & processing… this can take a while. Maybe go make a sandwich.');

  try {
    const res = await fetch('/process-video', {
      method: 'POST',
      body: form,
    });

    const data = await res.json();

    if (!res.ok) {
      const detail = data?.detail || res.statusText;
      throw new Error(detail);
    }

    setStatus('Success! Downloading annotated video.', 'success');
    resultCard.hidden = false;
    resultLink.href = data.url;
    resultVideo.src = data.url;
    resultVideo.load();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, 'error');
    resultCard.hidden = true;
  } finally {
    submitBtn.disabled = false;
  }
});
