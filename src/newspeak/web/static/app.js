const $ = id => document.getElementById(id);
const video = $('video'), cameraToggle = $('cameraToggle');
const messagesDiv = $('messages'), statusEl = $('status');
const stateDot = $('stateDot'), stateText = $('stateText');
const viewportWrap = $('viewportWrap');
const waveformCanvas = $('waveform');
const waveformCtx = waveformCanvas.getContext('2d');
const modeButton = $('modeButton');
const currentModeName = $('currentModeName');
const modeSheet = $('modeSheet');
const modeCards = $('modeCards');
const closeModeSheet = $('closeModeSheet');
const resetModal = $('resetModal');
const cancelResetBtn = $('cancelResetBtn');
const confirmResetBtn = $('confirmResetBtn');
const coachNotesList = $('coachNotes');
const deepDiveBtn = $('deepDiveBtn');
const cameraCaptionText = $('cameraCaptionText');

let ws, mediaStream, myvad;
let cameraEnabled = true;
let audioCtx, currentSource;
let state = 'loading';
let ignoreIncomingAudio = false;

let modes = [];
let activeModeId = '';
let pendingModeId = null;
let sessionHasActivity = false;

// Coach state
let lastTurnId = null;
let deepDivePending = false;

// Streaming audio playback state
let streamSampleRate = 24000;
let streamNextTime = 0;
let streamSources = [];
let streamTtsTime = null;

// Waveform visualizer
let analyser, micSource;
const BAR_COUNT = 36;
const BAR_GAP = 4;
let waveformRAF;
let ambientPhase = 0;

function initWaveformCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = rect.width * dpr;
  waveformCanvas.height = rect.height * dpr;
  waveformCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function getStateColor() {
  const colors = {
    listening: '#2fb579',
    processing: '#f5a623',
    speaking: '#4f7df3',
    loading: '#8c98aa',
  };
  return colors[state] || colors.loading;
}

function drawWaveform() {
  const w = waveformCanvas.getBoundingClientRect().width;
  const h = waveformCanvas.getBoundingClientRect().height;
  waveformCtx.clearRect(0, 0, w, h);

  const barWidth = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
  const color = getStateColor();
  waveformCtx.fillStyle = color;

  let dataArray = null;
  if (analyser) {
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
  }

  for (let i = 0; i < BAR_COUNT; i++) {
    let amplitude;
    if (dataArray) {
      const binIndex = Math.floor((i / BAR_COUNT) * dataArray.length * 0.55);
      amplitude = dataArray[binIndex] / 255;
    }

    if (!dataArray || amplitude < 0.02) {
      ambientPhase += 0.00012;
      const drift = Math.sin(ambientPhase * 4 + i * 0.42) * 0.5 + 0.5;
      amplitude = 0.04 + drift * 0.05;
    }

    const barH = Math.max(3, amplitude * (h - 6));
    const x = i * (barWidth + BAR_GAP);
    const y = (h - barH) / 2;

    waveformCtx.globalAlpha = 0.28 + amplitude * 0.62;
    waveformCtx.beginPath();
    const r = Math.min(barWidth / 2, barH / 2, 4);
    waveformCtx.roundRect(x, y, barWidth, barH, r);
    waveformCtx.fill();
  }

  waveformCtx.globalAlpha = 1;
  waveformRAF = requestAnimationFrame(drawWaveform);
}

function updateSpeakingGlow() {
  if (state !== 'speaking' || !analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const avg = sum / data.length / 255;
  const intensity = 0.25 + avg * 0.55;
  const spread = 24 + avg * 42;
  viewportWrap.querySelector('.viewport-glow').style.boxShadow =
    `inset 0 0 ${spread}px rgba(79,125,243,${intensity})`;
  requestAnimationFrame(updateSpeakingGlow);
}

function setState(newState) {
  state = newState;

  stateDot.className = `dot ${newState}`;
  const labels = {
    loading: 'Getting ready',
    listening: "I'm listening",
    processing: 'Thinking',
    speaking: 'Speaking',
  };
  stateText.textContent = labels[newState] || newState;
  viewportWrap.className = `viewport-wrap ${newState}`;

  if (newState !== 'speaking') {
    viewportWrap.querySelector('.viewport-glow').style.boxShadow = '';
  }

  const stateVars = {
    listening: ['#2fb579', 'rgba(47,181,121,0.16)'],
    processing: ['#f5a623', 'rgba(245,166,35,0.18)'],
    speaking: ['#4f7df3', 'rgba(79,125,243,0.16)'],
    loading: ['#8c98aa', 'rgba(140,152,170,0.14)'],
  };
  const [glow, glowDim] = stateVars[newState] || stateVars.loading;
  document.documentElement.style.setProperty('--glow', glow);
  document.documentElement.style.setProperty('--glow-dim', glowDim);

  if (newState === 'speaking') requestAnimationFrame(updateSpeakingGlow);

  if (myvad) {
    myvad.setOptions({ positiveSpeechThreshold: newState === 'speaking' ? 0.92 : 0.5 });
  }

  if (newState === 'listening' && mediaStream && audioCtx && analyser) {
    if (!micSource) {
      micSource = audioCtx.createMediaStreamSource(mediaStream);
    }
    try { micSource.connect(analyser); } catch {}
  } else if (micSource && newState !== 'listening') {
    try { micSource.disconnect(analyser); } catch {}
  }
}

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    setStatus('connected', 'Ready');
    if (state !== 'loading') setState('listening');
  };
  ws.onclose = () => {
    setStatus('disconnected', 'Reconnecting');
    setTimeout(connect, 2000);
  };
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'text') {
      if (msg.transcription) {
        const userMsgs = messagesDiv.querySelectorAll('.msg.user');
        const lastUserMsg = userMsgs[userMsgs.length - 1];
        if (lastUserMsg) {
          const meta = lastUserMsg.querySelector('.meta');
          lastUserMsg.innerHTML = `${escapeHtml(msg.transcription)}${meta ? meta.outerHTML : ''}`;
        }
      }
      addMessage('assistant', escapeHtml(msg.text));
      if (msg.turn_id) {
        lastTurnId = msg.turn_id;
        deepDiveBtn.disabled = false;
      }
    } else if (msg.type === 'audio_start') {
      if (ignoreIncomingAudio) return;
      streamSampleRate = msg.sample_rate || 24000;
      startStreamPlayback();
    } else if (msg.type === 'audio_chunk') {
      if (ignoreIncomingAudio) return;
      queueAudioChunk(msg.audio);
    } else if (msg.type === 'audio_end') {
      if (ignoreIncomingAudio) {
        ignoreIncomingAudio = false;
        stopPlayback();
        setState('listening');
        return;
      }
      streamTtsTime = msg.tts_time;
    } else if (msg.type === 'coach') {
      handleCoachEvent(msg);
    } else if (msg.type === 'modes_list') {
      populateModeCards(msg.modes, msg.active_mode_id);
    } else if (msg.type === 'mode_changed') {
      ignoreIncomingAudio = false;
      setActiveMode(msg.active_mode_id);
      setState('listening');
      setStatus('connected', 'Ready');
    } else if (msg.type === 'error') {
      if (msg.code === 'no_turn' || msg.code === 'turn_not_found') {
        deepDivePending = false;
        deepDiveBtn.classList.remove('loading');
        deepDiveBtn.disabled = lastTurnId === null;
        return;
      }
      addMessage('assistant', escapeHtml(msg.message || 'Something went wrong.'), msg.code || 'error');
      setState('listening');
      setStatus('connected', 'Ready');
    }
  };
}

function handleCoachEvent(msg) {
  if (deepDivePending && msg.deep) {
    deepDivePending = false;
    deepDiveBtn.classList.remove('loading');
    deepDiveBtn.disabled = false;
  }
  if (msg.skipped) return;

  sessionHasActivity = true;
  const empty = coachNotesList.querySelector('.coach-empty');
  if (empty) empty.remove();

  const li = document.createElement('li');
  li.className = `coach-note${msg.deep ? ' deep' : ''}`;

  const meta = document.createElement('div');
  meta.className = 'coach-meta';
  meta.textContent = msg.deep ? 'Deep dive' : 'Quick tip';

  const text = document.createElement('div');
  text.innerHTML = escapeHtml(msg.note);

  li.append(meta, text);
  coachNotesList.appendChild(li);
  coachNotesList.scrollTop = coachNotesList.scrollHeight;
}

function populateModeCards(nextModes, activeId) {
  modes = nextModes || [];
  modeCards.innerHTML = '';

  if (modes.length === 0) {
    modeButton.style.display = 'none';
    return;
  }

  modeButton.style.display = '';
  modes.forEach(mode => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'mode-card';
    card.dataset.modeId = mode.id;

    const title = document.createElement('strong');
    title.textContent = mode.name;

    const description = document.createElement('p');
    description.textContent = mode.description || 'Practice a fresh conversation.';

    const tag = document.createElement('span');
    tag.className = 'mode-tag';
    tag.textContent = mode.target_language || mode.coach_language || 'Practice';

    card.append(title, description, tag);
    card.addEventListener('click', () => chooseMode(mode.id));
    modeCards.appendChild(card);
  });

  setActiveMode(activeId || activeModeId || modes[0].id);
}

function setActiveMode(modeId) {
  activeModeId = modeId || activeModeId;
  const activeMode = modes.find(mode => mode.id === activeModeId);
  currentModeName.textContent = activeMode ? activeMode.name : 'Choose mode';
  document.querySelectorAll('.mode-card').forEach(card => {
    card.classList.toggle('active', card.dataset.modeId === activeModeId);
  });
}

function chooseMode(modeId) {
  if (!modeId || modeId === activeModeId) {
    closePracticePicker();
    return;
  }

  if (sessionNeedsReset()) {
    pendingModeId = modeId;
    resetModal.hidden = false;
    return;
  }

  switchMode(modeId, { resetUi: false });
}

function sessionNeedsReset() {
  return sessionHasActivity
    || messagesDiv.children.length > 0
    || coachNotesList.querySelector('.coach-note') !== null
    || deepDivePending
    || state === 'speaking'
    || streamSources.length > 0;
}

function switchMode(modeId, { resetUi }) {
  if (resetUi) clearLocalSession();
  ignoreIncomingAudio = true;
  stopPlayback();
  closePracticePicker();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_mode', mode_id: modeId }));
  } else {
    ignoreIncomingAudio = false;
    setActiveMode(modeId);
  }
}

function clearLocalSession() {
  messagesDiv.innerHTML = '';
  resetCoachNotes();
  lastTurnId = null;
  deepDivePending = false;
  deepDiveBtn.classList.remove('loading');
  deepDiveBtn.disabled = true;
  sessionHasActivity = false;
  setState('listening');
  setStatus('connected', 'Ready');
}

function resetCoachNotes() {
  coachNotesList.innerHTML = '';
  const placeholder = document.createElement('li');
  placeholder.className = 'coach-empty';
  placeholder.textContent = 'A quick tip will appear here after you speak.';
  coachNotesList.appendChild(placeholder);
}

function openPracticePicker() {
  modeSheet.hidden = false;
  modeButton.setAttribute('aria-expanded', 'true');
}

function closePracticePicker() {
  modeSheet.hidden = true;
  modeButton.setAttribute('aria-expanded', 'false');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}

modeButton.addEventListener('click', openPracticePicker);
closeModeSheet.addEventListener('click', closePracticePicker);
modeSheet.addEventListener('click', event => {
  if (event.target.matches('[data-close-mode-sheet]')) closePracticePicker();
});

cancelResetBtn.addEventListener('click', () => {
  pendingModeId = null;
  resetModal.hidden = true;
});

confirmResetBtn.addEventListener('click', () => {
  const modeId = pendingModeId;
  pendingModeId = null;
  resetModal.hidden = true;
  if (modeId) switchMode(modeId, { resetUi: true });
});

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    resetModal.hidden = true;
    pendingModeId = null;
    closePracticePicker();
  }
});

deepDiveBtn.addEventListener('click', () => {
  if (!lastTurnId || deepDivePending) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  sessionHasActivity = true;
  deepDivePending = true;
  deepDiveBtn.classList.add('loading');
  deepDiveBtn.disabled = true;
  ws.send(JSON.stringify({ type: 'deep_dive', turn_id: lastTurnId }));
});

function setStatus(cls, text) {
  statusEl.className = `status-pill ${cls}`;
  statusEl.textContent = text;
}

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    video.srcObject = mediaStream;
    cameraCaptionText.textContent = 'Camera ready';
    return;
  } catch (e) { console.warn('Video+audio failed:', e.message); }

  const streams = await Promise.allSettled([
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } }),
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }),
  ]);
  mediaStream = new MediaStream();
  streams.forEach(r => { if (r.status === 'fulfilled') r.value.getTracks().forEach(t => mediaStream.addTrack(t)); });
  if (mediaStream.getVideoTracks().length) {
    video.srcObject = mediaStream;
    cameraCaptionText.textContent = 'Camera ready';
  }
  if (!mediaStream.getAudioTracks().length) {
    cameraEnabled = false;
    cameraCaptionText.textContent = 'Mic unavailable';
  }
}

function captureFrame() {
  if (!cameraEnabled || !video.videoWidth) return null;
  const canvas = document.createElement('canvas');
  const scale = 320 / video.videoWidth;
  canvas.width = 320;
  canvas.height = video.videoHeight * scale;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
}

let speakingStartedAt = 0;
const BARGE_IN_GRACE_MS = 800;

function handleSpeechStart() {
  if (state === 'speaking') {
    if (Date.now() - speakingStartedAt < BARGE_IN_GRACE_MS) {
      console.log('Barge-in suppressed during echo grace period');
      return;
    }
    stopPlayback();
    ignoreIncomingAudio = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    setState('listening');
    console.log('Barge-in interrupted playback');
  }
}

function handleSpeechEnd(audio) {
  if (state !== 'listening') return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const wavBase64 = float32ToWavBase64(audio);
  const imageBase64 = captureFrame();

  sessionHasActivity = true;
  setState('processing');
  setStatus('processing', 'Thinking');
  addMessage(
    'user',
    '<span class="loading-dots"><span></span><span></span><span></span></span>',
    imageBase64 ? 'Camera included' : ''
  );

  const payload = { audio: wavBase64 };
  if (imageBase64) payload.image = imageBase64;
  ws.send(JSON.stringify(payload));
}

function float32ToWavBase64(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8,'WAVE'); w(12,'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36,'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function stopPlayback() {
  for (const src of streamSources) {
    try { src.stop(); } catch {}
  }
  streamSources = [];
  currentSource = null;
  streamNextTime = 0;
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
  }
}

function startStreamPlayback() {
  stopPlayback();
  ensureAudioCtx();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  streamNextTime = audioCtx.currentTime + 0.05;
  speakingStartedAt = Date.now();
  setState('speaking');
}

function queueAudioChunk(base64Pcm) {
  ensureAudioCtx();

  const bin = atob(base64Pcm);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const audioBuffer = audioCtx.createBuffer(1, float32.length, streamSampleRate);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.connect(analyser);

  const startAt = Math.max(streamNextTime, audioCtx.currentTime);
  source.start(startAt);
  streamNextTime = startAt + audioBuffer.duration;

  streamSources.push(source);
  currentSource = source;

  source.onended = () => {
    const idx = streamSources.indexOf(source);
    if (idx !== -1) streamSources.splice(idx, 1);
    if (streamSources.length === 0 && state === 'speaking') {
      currentSource = null;
      setState('listening');
      setStatus('connected', 'Ready');
    }
  };
}

function addMessage(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `${text}${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ''}`;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  sessionHasActivity = true;
}

cameraToggle.addEventListener('click', () => {
  cameraEnabled = !cameraEnabled;
  cameraToggle.classList.toggle('active', cameraEnabled);
  cameraToggle.textContent = cameraEnabled ? 'Camera On' : 'Camera Off';
  cameraCaptionText.textContent = cameraEnabled ? 'Camera ready' : 'Camera paused';
  video.style.opacity = cameraEnabled ? 1 : 0.35;
});

async function init() {
  initWaveformCanvas();
  window.addEventListener('resize', initWaveformCanvas);

  resetCoachNotes();

  try {
    const resp = await fetch('/modes');
    if (resp.ok) {
      const data = await resp.json();
      populateModeCards(data.modes, data.active_mode_id);
    }
  } catch {}

  await startCamera();
  connect();

  myvad = await vad.MicVAD.new({
    getStream: async () => new MediaStream(mediaStream.getAudioTracks()),
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.25,
    redemptionMs: 1000,
    minSpeechMs: 300,
    preSpeechPadMs: 300,
    onSpeechStart: handleSpeechStart,
    onSpeechEnd: handleSpeechEnd,
    onVADMisfire: () => { console.log('VAD misfire, too short'); },
    onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
    baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
  });

  myvad.start();

  const initAudio = () => {
    ensureAudioCtx();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    document.removeEventListener('click', initAudio);
    document.removeEventListener('keydown', initAudio);
  };
  document.addEventListener('click', initAudio);
  document.addEventListener('keydown', initAudio);
  ensureAudioCtx();

  setState('listening');
  drawWaveform();

  console.log('VAD initialized and listening');
}

init();
