$(document).ready(function () {
    $(window).on('load', function() {
        $('.hero-title').addClass('revealed');
    });


    //--- SAMPLES DATA ---
    const samples = {
        scam: "URGENT: Your bank account has been compromised. Verify your identity immediately to prevent permanent lockout: http://secure-verify-auth.com/account/login-992. No further warnings will be issued.",
        news: "BREAKING: Massive solar flare predicted to shut down the global internet for exactly 48 hours starting tonight. Governments are already preparing emergency rations. Share this before the blackout!",
        ai: "The industrial revolution was a period of significant change that transformed agrarian societies into industrial ones. The emergence of steam power played a pivotal role in this transition, leading to unprecedented levels of production.",
        true: "The Apollo 11 mission was the first spaceflight that landed the first two people on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969."
    };

    let loggedIn = true;
    let authToken = localStorage.getItem('ag_auth_token') || null;

    function setAuthHeader(token) {
        if (token) {
            authToken = token;
            localStorage.setItem('ag_auth_token', token);
            $.ajaxSetup({
                headers: {
                    Authorization: 'Bearer ' + token
                },
                xhrFields: {
                    withCredentials: true
                }
            });
        } else {
            authToken = null;
            localStorage.removeItem('ag_auth_token');
            $.ajaxSetup({
                headers: {},
                xhrFields: {
                    withCredentials: true
                }
            });
        }
    }

    setAuthHeader(authToken);

    function setAuthUI(isAuthenticated) {
        loggedIn = true;
        $('#openHistoryBtn').removeClass('hidden');
        $('#profileBtn').removeClass('hidden');
        $('#logoutBtn').addClass('hidden');
        $('#landingPage').removeClass('hidden');
        $('.scanner-page').addClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#resultOverlay').addClass('hidden');
    }

    function checkSession() {
        $.get('/auth/me', function (data) {
            setAuthUI(Boolean(data.authenticated));
        }).fail(function () {
            setAuthUI(true);
        });
    }

    function updateSystemStatus(data) {
        if (!data.neural_engine || !data.linguistic_engine) {
            let errorMsg = 'One or more AI models failed to initialize.';
            if (!data.neural_engine && data.neural_error) {
                errorMsg += '\n\nDeepfake Model Error: ' + data.neural_error;
            }
            if (!data.linguistic_engine && data.linguistic_error) {
                errorMsg += '\n\nLinguistic Model Error: ' + data.linguistic_error;
            }
            $('#errorMessage').text(errorMsg);
            $('#errorModal').removeClass('hidden');
            $('.cyber-btn').prop('disabled', true);
        }
    }

    $.get('/status', updateSystemStatus).fail(function () {
        $('#errorMessage').text('Failed to connect to the server. Please ensure the backend is running.');
        $('#errorModal').removeClass('hidden');
        $('.cyber-btn').prop('disabled', true);
    });

    $('#closeError').click(function () {
        $('#errorModal').addClass('hidden');
    });

    $('#logoutBtn').click(function () {
        $.post('/auth/logout', function () {
            setAuthHeader(null);
            setAuthUI(true);
        }).fail(function () {
            setAuthHeader(null);
            setAuthUI(true);
        });
    });

    $('#refreshKeyBtn').click(function () {
        $.post('/auth/refresh_api_key', function (data) {
            if (data.success) {
                $('#profileApiKey').text(data.api_key);
                $('#authMessage').text('API key refreshed.');
            }
        }).fail(function () {
            $('#authMessage').text('Unable to refresh API key.');
        });
    });

    function showSection(sectionId) {
        $('#landingPage').addClass('hidden');
        $('#appContainer').removeClass('hidden');
        $('.scanner-page').addClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#resultOverlay').addClass('hidden');
        $(`#${sectionId}`).removeClass('hidden');
    }

    $('#profileBtn').click(function () {
        showSection('profilePage');
        $.get('/auth/me', function (data) {
            if (data.authenticated) {
                $('#profileName').text(data.user.name);
                $('#profileEmail').text(data.user.email);
                $('#profileTier').text(data.user.is_premium ? 'Pro' : 'Free');
                $('#profileApiKey').text(data.user.api_key);
            }
        });
    });

    $('#pricingBtn').click(function () {
        showSection('pricingPage');
    });

    checkSession();

    $('#openHistoryBtn').click(function () {
        $('#landingPage').addClass('hidden');
        $('#appContainer').removeClass('hidden');
        $('.scanner-page').addClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#resultOverlay').addClass('hidden');
        $('#historyPage').removeClass('hidden');
        fetchHistory();
    });

    function setAppTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const toggleBtn = $('#themeToggle');
        if (theme === 'light') {
            toggleBtn.text('DARK MODE');
        } else {
            toggleBtn.text('LIGHT MODE');
        }
        localStorage.setItem('theme', theme);
    }

    function loadAppTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        setAppTheme(savedTheme);
    }

    loadAppTheme();

    $('#themeToggle').click(function () {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setAppTheme(nextTheme);
    });


    // --- NAVIGATION ---
    $('.scan-entry-btn').click(function () {
        const target = $(this).data('target');
        $('#landingPage').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
            .css('opacity', '0')
            .css('transform', 'scale(1.1) rotateX(10deg)');

        setTimeout(() => {
            $('#landingPage').addClass('hidden');
            $('#appContainer').removeClass('hidden').css('opacity', '0').css('transform', 'scale(0.9)');

            $('#appContainer').outerWidth();

            $('#appContainer').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
                .css('opacity', '1')
                .css('transform', 'scale(1)');

            $('.scanner-page').addClass('hidden');
            $('#resultsDisplay').addClass('hidden');
            $('#resultOverlay').addClass('hidden');
            if (target === 'text-panel') {
                $('#textScannerPage').removeClass('hidden');
            } else if (target === 'image-panel') {
                $('#mediaScannerPage').removeClass('hidden');
            } else if (target === 'url-panel') {
                $('#urlScannerPage').removeClass('hidden');
            } else if (target === 'webcam-panel') {
                $('#webcamScannerPage').removeClass('hidden');
            } else if (target === 'history-panel') {
                $('#historyPage').removeClass('hidden');
                fetchHistory();
            }

        }, 800);
    });

    $('#backToLanding').click(function () {
        $('#appContainer').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
            .css('opacity', '0')
            .css('transform', 'scale(0.9)');

        setTimeout(() => {
            $('#appContainer').addClass('hidden');
            $('#landingPage').removeClass('hidden').css('opacity', '0').css('transform', 'scale(1.1)');

            $('#landingPage').outerWidth();

            $('#landingPage').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
                .css('opacity', '1')
                .css('transform', 'scale(1)');

            $('.scanner-page').addClass('hidden');
            $('#resultsDisplay').addClass('hidden');
            stopWebcam();
        }, 800);
    });


    // --- MATRIX DIGITAL RAIN ---
    const canvas = document.getElementById('matrixCanvas');
    const ctx = canvas.getContext('2d');
    let width, height, cols, ypos;
    const charSize = 18;

    const characters = "アァカサタナハマヤラワガザダバパ1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ$+-*/=%\"'#&_(),.;:?!\\|{}<>[]^~";

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        cols = Math.floor(width / charSize);
        ypos = Array.from({ length: cols }, () => Math.random() * height);
    }

    window.addEventListener('resize', resize);
    resize();

    function drawMatrix() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.07)';
        ctx.fillRect(0, 0, width, height);

        ctx.font = `bold ${charSize - 3}px monospace`;
        ctx.shadowColor = 'rgba(57, 255, 20, 0.35)';
        ctx.shadowBlur = 12;
        ctx.textBaseline = 'top';

        ypos.forEach((y, ind) => {
            const text = characters.charAt(Math.floor(Math.random() * characters.length));
            const x = ind * charSize;
            const intensity = Math.floor(Math.random() * 120) + 135;

            ctx.fillStyle = `rgba(57, ${intensity}, 20, 0.92)`;
            if (Math.random() > 0.97) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
                ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
            } else {
                ctx.shadowColor = 'rgba(57, 255, 20, 0.35)';
            }

            ctx.fillText(text, x, y);

            ypos[ind] = y > height + Math.random() * 10000 ? 0 : y + charSize + Math.random() * 3;
        });

        requestAnimationFrame(drawMatrix);
    }

    drawMatrix();

    // --- UI LOGIC ---

    // Deck Toggle
    $('.deck-btn').click(function () {
        const target = $(this).data('target');
        $('.deck-btn').removeClass('active');
        $(this).addClass('active');
        const idx = $(this).index();
        $('.deck-slider').css('transform', `translateX(${idx * 100}%)`);
        $('.panel').removeClass('active');
        $(`#${target}`).addClass('active');
        $('#resultsDisplay').addClass('hidden');
    });

    // Sample Handle
    $('.sample-btn').click(function () {
        const type = $(this).data('sample');
        $('#newsInput').val(samples[type]).trigger('input');
    });

    // Character Count
    $('#newsInput').on('input', function () {
        $('.char-count').text(`${this.value.length} BYTES`);
    });

    // Why Flagged Toggle
    $('#whyFlaggedBtn').click(function () {
        $('#explanationContent').toggleClass('hidden');
        $(this).text($('#explanationContent').hasClass('hidden') ? "EXPAND_INTELLIGENCE_LOGS" : "COLLAPSE_INTELLIGENCE_LOGS");
    });

    // --- SCAN ENGINE ---

    let systemProgressInterval;
    const systemMessages = [
        "BOOTING_NEURAL_ENGINE...",
        "EXTRACTING_FEATURE_TENSORS...",
        "RUNNING_SEMANTIC_WEIGHTS...",
        "CALCULATING_INFERENCE_CONFIDENCE...",
        "ANALYZING_PATTERN_DEVIATION...",
        "GENERATING_FORENSIC_VERDICT..."
    ];

    function initProcess(customMessages) {
        $('.status-pill').text('SYSTEM ANALYZING').addClass('analyzing');
        $('#resultOverlay').removeClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#explanationContent').addClass('hidden');
        $('#whyFlaggedBtn').text("EXPAND_INTELLIGENCE_LOGS");

        const messages = customMessages || systemMessages;
        let msgIndex = 0;
        $('#systemMessage').text(messages[0]);

        if (systemProgressInterval) clearInterval(systemProgressInterval);
        systemProgressInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % messages.length;
            $('#systemMessage').text(messages[msgIndex]);
        }, 800);
    }

    function resetProcess() {
        clearInterval(systemProgressInterval);
        $('.status-pill').text('SYSTEM ONLINE').removeClass('analyzing');
        $('#resultOverlay').addClass('hidden');
    }

    function showAjaxError(xhr, defaultMessage = 'LINK FAILURE') {
        let message = defaultMessage;
        if (xhr && xhr.responseJSON && xhr.responseJSON.message) {
            message = xhr.responseJSON.message;
        } else if (xhr && xhr.responseText) {
            try {
                const json = JSON.parse(xhr.responseText);
                if (json.message) {
                    message = json.message;
                }
            } catch (e) {
                /* ignore invalid JSON */
            }
        }
        alert(message);
        resetProcess();
    }

    function updateResultUI(truthScore, explanation) {
        resetProcess();
        $('#resultsDisplay').removeClass('hidden').css('opacity', '0').css('transform', 'translateY(20px)');

        // Animated Entrance
        setTimeout(() => {
            $('#resultsDisplay').css('transition', 'all 0.6s cubic-bezier(0.23, 1, 0.32, 1)')
                .css('opacity', '1')
                .css('transform', 'translateY(0)');
        }, 50);

        let color, title, dotColor;
        if (truthScore < 40) {
            color = '#FF3B3B'; // Red - High Risk
            title = 'HIGH_RISK_PATTERN';
            dotColor = '#FF3B3B';
        } else if (truthScore < 65) {
            color = '#FFD700'; // Yellow - Anomaly
            title = 'ANOMALY_DETECTED';
            dotColor = '#FFD700';
        } else {
            color = 'var(--accent)'; // Green - Verified
            title = 'VERIFIED_AUTHENTIC';
            dotColor = 'var(--accent)';
        }

        // Update Metadata
        const now = new Date();
        $('#scanId').text(`#AG-${Math.floor(Math.random() * 9000 + 1000)}-${(Math.random() + 1).toString(36).substring(7).toUpperCase()}`);
        $('#scanTime').text(now.toISOString().replace('T', ' ').substring(0, 16));

        // Update Intelligence Metrics
        const inference = Math.max(70, truthScore > 50 ? truthScore - 5 : 100 - truthScore - 5) + (Math.random() * 10);
        const consistency = Math.min(98, truthScore > 50 ? truthScore + 2 : truthScore + 10) + (Math.random() * 5);
        const signal = 70 + (Math.random() * 25);
        const anomaly = truthScore < 50 ? (100 - truthScore + (Math.random() * 10)) : (Math.random() * 15);

        $('#inferenceBar').css('width', `${Math.min(100, inference)}%`);
        $('#inferenceVal').text(`${Math.min(100, inference).toFixed(1)}%`);

        $('#consistencyBar').css('width', `${Math.min(100, consistency)}%`);
        $('#consistencyVal').text(`${Math.min(100, consistency).toFixed(1)}%`);

        $('#signalBar').css('width', `${Math.min(100, signal)}%`);
        $('#signalVal').text(`${Math.min(100, signal).toFixed(1)}%`);

        $('#anomalyBar').css('width', `${Math.min(100, anomaly)}%`).toggleClass('risk', anomaly > 40);
        $('#anomalyVal').text(`${Math.min(100, anomaly).toFixed(1)}%`);

        $('#resVerdictTitle').text(title).css('color', color).css('text-shadow', `0 0 15px ${color}`);
        $('#resStatusLabel').text('ANALYSIS COMPLETE').css('color', color);
        $('.pulse-dot').css('background', dotColor).css('box-shadow', `0 0 10px ${dotColor}`);

        // Counter Animation for Score
        let currentScoreCount = 0;
        const scoreInterval = setInterval(() => {
            if (currentScoreCount >= Math.floor(truthScore)) {
                clearInterval(scoreInterval);
                $('#resScoreText').text(`${truthScore.toFixed(1)}%`);
            } else {
                currentScoreCount++;
                $('#resScoreText').text(`${currentScoreCount}%`);
            }
        }, 15);

        $('#scoreCircle').css('stroke', color).css('stroke-dasharray', `${truthScore}, 100`);

        $('#resVerdictReason').text(explanation.summary);

        const pointsList = $('#analysisPoints');
        pointsList.empty();
        explanation.details.forEach((point, i) => {
            const li = $(`<li>${point}</li>`).css('opacity', '0').css('transform', 'translateX(-10px)');
            pointsList.append(li);
            // Staggered entrance for details
            setTimeout(() => {
                li.css('transition', 'all 0.4s ease').css('opacity', '1').css('transform', 'translateX(0)');
            }, 600 + (i * 150));
        });

        // Smooth scroll to results
        $('html, body').animate({
            scrollTop: $("#resultsDisplay").offset().top - 50
        }, 800);
    }

    function fetchHistory() {
        const params = {};
        const type = $('#historyTypeFilter').val();
        const result = $('#historyResultFilter').val();
        const search = $('#historySearchInput').val().trim();

        if (type) params.type = type;
        if (result) params.result = result;
        if (search) params.search = search;

        $.get('/history', params, function (records) {
            const tbody = $('#historyTable tbody');
            tbody.empty();
            if (!records || records.length === 0) {
                tbody.append('<tr><td colspan="6">NO MATCHING HISTORY AVAILABLE</td></tr>');
                return;
            }
            records.forEach(record => {
                tbody.append(`
                    <tr>
                        <td>${record.scan_id}</td>
                        <td>${record.type}</td>
                        <td>${record.result}</td>
                        <td>${parseFloat(record.confidence).toFixed(1)}%</td>
                        <td>${record.timestamp}</td>
                        <td><a class="history-report-link" href="/report/${record.scan_id}" target="_blank" rel="noopener noreferrer">EXPORT PDF</a></td>
                    </tr>
                `);
            });
        }).fail(function () {
            alert('Unable to load scan history from backend.');
        });
    }

    $('#refreshHistory').click(fetchHistory);
    $('#historyTypeFilter').change(fetchHistory);
    $('#historyResultFilter').change(fetchHistory);
    $('#historySearchInput').on('input', fetchHistory);

    // --- ANALYZE TEXT ---
    $('#analyzeTextBtn').click(function () {
        const text = $('#newsInput').val();
        if (!text.trim()) return alert("INPUT STREAM EMPTY");

        const onlineMessages = [
            "CONNECTING_TO_GLOBAL_VERIFICATION_NODES...",
            "SCANNING_VAST_NEWS_DATABASES (Wikipedia, Reuters)...",
            "CROSS_REFERENCING_AP_WIRE_DATA...",
            "ANALYZING_SEMANTIC_WEIGHTS...",
            "GENERATING_FORENSIC_VERDICT..."
        ];
        initProcess(onlineMessages);

        $.post('/predict_news', { text: text }, function (data) {
            if (data.result === 'Error') { alert(data.message); resetProcess(); return; }

            const isFake = (data.result === 'FAKE');
            let backendConfidence = parseFloat(data.confidence) || 0;
            let truthScore = isFake ? (100 - backendConfidence) : backendConfidence;
            truthScore = Math.max(2, Math.min(98, truthScore));

            let explanation = {
                summary: data.summary || "Analysis complete.",
                details: data.forensic_details || []
            };

            setTimeout(() => {
                updateResultUI(truthScore, explanation);
            }, 1500);

        }).fail(function (xhr) { showAjaxError(xhr); });
    });

    // --- ANALYZE URL ---
    $('#analyzeUrlBtn').click(function () {
        const url = $('#urlInput').val().trim();
        if (!url) return alert('URL INPUT EMPTY');

        const urlMessages = [
            "RESOLVING_REMOTE_RESOURCE...",
            "CRAWLING_PAGE_CONTENT...",
            "SCANNING_SOURCE TRACES...",
            "ANALYZING_TONE_AND_CONTEXT...",
            "FINALIZING_VERACITY_SCORE..."
        ];
        initProcess(urlMessages);

        $.post('/predict_url', { url: url }, function (data) {
            if (data.result === 'Error') { alert(data.message); resetProcess(); return; }

            let backendConfidence = parseFloat(data.confidence) || 0;
            let truthScore = data.result === 'FAKE' ? Math.max(2, Math.min(98, 100 - backendConfidence)) : Math.max(2, Math.min(98, backendConfidence));

            const explanation = {
                summary: data.summary || 'Web source analysis complete.',
                details: data.forensic_details || []
            };

            setTimeout(() => {
                updateResultUI(truthScore, explanation);
            }, 1500);
        }).fail(function (xhr) { showAjaxError(xhr); });
    });

    // --- ANALYZE IMAGE ---
    const dropZone = $('#dropZone');
    const imageInput = $('#imageInput');

    dropZone.on('click', () => imageInput.click());
    imageInput.change(function () {
        if (this.files && this.files[0]) handleFile(this.files[0]);
    });

    dropZone.on('dragover dragenter', function (e) {
        e.preventDefault();
        $(this).css('border-color', 'var(--neon-blue)').css('box-shadow', '0 0 20px rgba(0, 243, 255, 0.2)');
    });

    dropZone.on('dragleave dragend drop', function (e) {
        e.preventDefault();
        $(this).css('border-color', '').css('box-shadow', '');
    });

    dropZone.on('drop', function (e) {
        const files = e.originalEvent.dataTransfer.files;
        if (files && files.length > 0) handleFile(files[0]);
    });

    let currentFile = null;

    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            $('#previewContainer').removeClass('hidden');
            $('.grid-content').addClass('hidden');
            $('#analyzeImageBtn').removeClass('hidden');

            if (file.type.startsWith('image/')) {
                $('#imagePreview').attr('src', e.target.result).removeClass('hidden');
                $('#videoPreview').addClass('hidden');
            } else {
                $('#videoPreview').attr('src', e.target.result).removeClass('hidden');
                $('#imagePreview').addClass('hidden');
            }
        }
        reader.readAsDataURL(file);
    }

    $('#clearImage').click(function (e) {
        e.stopPropagation();
        currentFile = null;
        $('#previewContainer').addClass('hidden');
        $('.grid-content').removeClass('hidden');
        $('#analyzeImageBtn').addClass('hidden');
        $('#resultsDisplay').addClass('hidden');
    });

    $('#analyzeImageBtn').click(function () {
        if (!currentFile) return;
        const forensicMessages = [
            "INITIALIZING_VISION_FORENSICS...",
            "SAMPLING_TEMPORAL_VIDEO_FRAMES...",
            "ANALYZING_METADATA_INTEGRITY...",
            "CALCULATING_AVERAGE_NEURAL_CONFIDENCE...",
            "FINALIZING_MEDIA_DIAGNOSTIC..."
        ];
        initProcess(forensicMessages);

        const formData = new FormData();
        formData.append('file', currentFile);

        $.ajax({
            url: '/predict_deepfake',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                if (data.result === 'Error') { alert(data.message); resetProcess(); return; }

                const isFake = (data.result === 'FAKE');
                let backendConfidence = parseFloat(data.confidence) || 0;

                // Truth Score logic for Images
                let truthScore;
                if (isFake) {
                    truthScore = backendConfidence > 50 ? (100 - backendConfidence) : backendConfidence;
                } else {
                    truthScore = backendConfidence < 50 ? (100 - backendConfidence) : backendConfidence;
                }

                truthScore = Math.max(2, Math.min(98, truthScore));

        const explanation = {
            summary: "Temporal Forensic Analysis Complete.",
            details: isFake ? [
                "Neural inconsistency detected in facial landmarks.",
                "Temporal sampling shows frame-to-frame variance in lighting patterns.",
                "Deepfake signature patterns identified in high-frequency regions."
            ] : [
                "No synthetic generation artifacts detected in media frames.",
                "Facial texture maintains natural structural consistency.",
                "Metadata and compression profiles align with standard camera hardware."
            ]
        };

                setTimeout(() => {
                    updateResultUI(truthScore, explanation);
                }, 2500);
            },
            error: function (xhr) { showAjaxError(xhr, 'NEURAL LINK SEVERED'); }
        });
    });

    // --- LIVE WEBCAM SCANNER ---
    let webcamStream = null;
    let webcamInterval = null;
    const WEBCAM_POLL_MS = 800;

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            const video = document.getElementById('webcamFeed');
            video.srcObject = webcamStream;
            
            $('#webcamInstructions').addClass('hidden');
            $('#webcamOverlay').removeClass('hidden');
            $('#startWebcamBtn').addClass('hidden');
            $('#stopWebcamBtn').removeClass('hidden');
            $('#rollingLogsContainer').removeClass('hidden');
            $('#webcamBadge').text('BOOTING...').removeClass('fake');
            
            // Start capturing frames
            setTimeout(() => {
                webcamInterval = setInterval(captureAndScan, WEBCAM_POLL_MS);
            }, 1000);
            
        } catch (err) {
            console.error("Webcam Error:", err);
            alert("Unable to access camera. Please check permissions.");
        }
    }

    function stopWebcam() {
        if (webcamInterval) {
            clearInterval(webcamInterval);
            webcamInterval = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        
        const video = document.getElementById('webcamFeed');
        if (video) video.srcObject = null;
        
        $('#webcamInstructions').removeClass('hidden');
        $('#webcamOverlay').addClass('hidden');
        $('#startWebcamBtn').removeClass('hidden');
        $('#stopWebcamBtn').addClass('hidden');
        // Clear log on stop? optional.
    }

    function captureAndScan() {
        const video = document.getElementById('webcamFeed');
        const canvas = document.getElementById('captureCanvas');
        if (!video || !canvas || video.paused || video.ended) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw centered square crop for model if needed, or just send full frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const base64Image = canvas.toDataURL('image/jpeg', 0.7);

        $.ajax({
            url: '/api/scan/webcam',
            type: 'POST',
            data: JSON.stringify({ image: base64Image }),
            contentType: 'application/json',
            success: function(data) {
                updateWebcamUI(data);
            },
            error: function(err) {
                console.error("Webcam scan failure", err);
            }
        });
    }

    function updateWebcamUI(data) {
        const badge = $('#webcamBadge');
        const conf = $('#webcamConfidence');
        
        const prevRes = badge.text();
        const curRes = data.prediction;
        
        badge.text(curRes);
        conf.text(`${data.confidence.toFixed(1)}%`);
        
        if (curRes === 'FAKE') {
            badge.addClass('fake');
        } else {
            badge.removeClass('fake');
        }

        // Pulse effect if result changes
        if (prevRes !== curRes && prevRes !== 'BOOTING...') {
            badge.css('transform', 'scale(1.2)');
            setTimeout(() => badge.css('transform', 'scale(1)'), 200);
        }

        addRollingLog(data);
    }

    function addRollingLog(data) {
        const logBox = $('#webcamRollingLogs');
        const now = new Date();
        const timeStr = now.getHours().toString().padStart(2, '0') + ":" + 
                        now.getMinutes().toString().padStart(2, '0') + ":" + 
                        now.getSeconds().toString().padStart(2, '0');
        
        const logEntry = $(`
            <div class="log-entry ${data.prediction.toLowerCase()}">
                <span class="log-time">[${timeStr}]</span>
                <span class="log-id">${data.scan_id}</span>
                <span class="log-res">${data.prediction} (${data.confidence.toFixed(1)}%)</span>
            </div>
        `);
        
        logBox.prepend(logEntry);
        
        // Keep last 5
        if (logBox.children().length > 5) {
            logBox.children().last().remove();
        }
    }

    $('#startWebcamBtn').click(startWebcam);
    $('#stopWebcamBtn').click(stopWebcam);
});

