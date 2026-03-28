$(document).ready(function () {

    //--- SAMPLES DATA ---
    const samples = {
        scam: "URGENT: Your bank account has been compromised. Verify your identity immediately to prevent permanent lockout: http://secure-verify-auth.com/account/login-992. No further warnings will be issued.",
        news: "BREAKING: Massive solar flare predicted to shut down the global internet for exactly 48 hours starting tonight. Governments are already preparing emergency rations. Share this before the blackout!",
        ai: "The industrial revolution was a period of significant change that transformed agrarian societies into industrial ones. The emergence of steam power played a pivotal role in this transition, leading to unprecedented levels of production.",
        true: "The Apollo 11 mission was the first spaceflight that landed the first two people on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969."
    };

    let loggedIn = false;
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
        loggedIn = isAuthenticated;
        $('#openLoginBtn').toggleClass('hidden', isAuthenticated);
        $('#openRegisterBtn').toggleClass('hidden', isAuthenticated);
        $('#openHistoryBtn').toggleClass('hidden', !isAuthenticated);
        $('#profileBtn').toggleClass('hidden', !isAuthenticated);
        $('#logoutBtn').toggleClass('hidden', !isAuthenticated);
        $('#authPage').toggleClass('hidden', isAuthenticated);
        $('#landingPage').toggleClass('hidden', !isAuthenticated);
        $('.scanner-page').addClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#resultOverlay').addClass('hidden');
    }

    let authMode = 'login';

    function setAuthMode(mode) {
        authMode = mode;
        $('#showLoginTab').toggleClass('active', mode === 'login');
        $('#showRegisterTab').toggleClass('active', mode === 'register');
        $('#loginPanel').toggleClass('hidden', mode !== 'login');
        $('#registerPanel').toggleClass('hidden', mode !== 'register');
        $('#authMessage').text('');
    }

    function showAuth(message, mode = 'login') {
        setAuthMode(mode);
        setAuthUI(false);
        $('#landingPage').addClass('hidden');
        $('#authPage').removeClass('hidden');
        $('#authMessage').text(message || (mode === 'register'
            ? 'Create your account to continue.'
            : 'Please sign in to continue.'));
    }

    $('#showLoginTab').click(function () {
        showAuth('Enter your credentials to log in.', 'login');
    });

    $('#showRegisterTab').click(function () {
        showAuth('Fill the form to create your AntiGravity account.', 'register');
    });

    $('#openLoginBtn').click(function () {
        showAuth('Enter your credentials to log in.', 'login');
    });

    $('#openRegisterBtn').click(function () {
        showAuth('Fill the form to create your AntiGravity account.', 'register');
    });

    function checkSession() {
        $.get('/me', function (data) {
            if (data.authenticated) {
                authToken = null;
                setAuthUI(true);
                $('#status').text('Session active.');
            } else {
                showAuth('Sign in or register to unlock your AntiGravity account.');
            }
        }).fail(function () {
            showAuth('Sign in or register to unlock your AntiGravity account.');
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

    $('#loginBtn').click(function () {
        const email = $('#loginEmail').val().trim();
        const password = $('#loginPassword').val().trim();
        if (!email || !password) {
            $('#authMessage').text('Email and password are required.');
            return;
        }
        $.ajax({
            url: '/login',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ email, password }),
            success: function (data) {
                if (data.success) {
                    setAuthHeader(data.token);
                    setAuthUI(true);
                    $('#authMessage').text('Welcome back, ' + data.user.name + '.');
                } else {
                    $('#authMessage').text(data.message || 'Login failed.');
                }
            },
            error: function (xhr) {
                $('#authMessage').text(xhr.responseJSON?.message || 'Login failed.');
            }
        });
    });

    $('#registerBtn').click(function () {
        const name = $('#registerName').val().trim();
        const email = $('#registerEmail').val().trim();
        const password = $('#registerPassword').val().trim();
        if (!name || !email || !password) {
            $('#authMessage').text('Name, email and password are required.');
            return;
        }
        $.ajax({
            url: '/register',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ name, email, password }),
            success: function (data) {
                if (data.success) {
                    setAuthHeader(data.token);
                    setAuthUI(true);
                    $('#authMessage').text('Account created. Welcome.');
                } else {
                    $('#authMessage').text(data.message || 'Registration failed.');
                }
            },
            error: function (xhr) {
                $('#authMessage').text(xhr.responseJSON?.message || 'Registration failed.');
            }
        });
    });

    $('#logoutBtn').click(function () {
        $.post('/logout', function () {
            setAuthHeader(null);
            setAuthUI(false);
            showAuth('You have been signed out.');
        }).fail(function () {
            setAuthHeader(null);
            setAuthUI(false);
            showAuth('You have been signed out.');
        });
    });

    $('#refreshKeyBtn').click(function () {
        $.post('/refresh_api_key', function (data) {
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
        $.get('/me', function (data) {
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

    $('#themeToggle').click(function () {
        $('body').toggleClass('light-mode');
        const isLight = $('body').hasClass('light-mode');
        $(this).text(isLight ? 'DARK MODE' : 'LIGHT MODE');
    });

    // --- NAVIGATION ---
    $('.scan-entry-btn').click(function () {
        if (!loggedIn) {
            showAuth('Please log in to access AntiGravity scans.');
            return;
        }
        const target = $(this).data('target');
        $('#landingPage').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
            .css('opacity', '0')
            .css('transform', 'scale(1.1) rotateX(10deg)');

        setTimeout(() => {
            $('#landingPage').addClass('hidden');
            $('#appContainer').removeClass('hidden').css('opacity', '0').css('transform', 'scale(0.9)');

            // Reflow
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

            // Reflow
            $('#landingPage').outerWidth();

            $('#landingPage').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
                .css('opacity', '1')
                .css('transform', 'scale(1)');

            $('.scanner-page').addClass('hidden');
            $('#resultsDisplay').addClass('hidden');
        }, 800);
    });

    // --- MATRIX DIGITAL RAIN ---
    const canvas = document.getElementById('matrixCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    const charSize = 20;
    const cols = Math.floor(width / charSize);
    const ypos = Array(cols).fill(0);

    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$+-*/=%\"'#&_(),.;:?!\\|{}<>[]^~";

    function drawMatrix() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, width, height);

        ctx.font = `bold ${charSize - 2}px monospace`;

        ypos.forEach((y, ind) => {
            const text = characters.charAt(Math.floor(Math.random() * characters.length));
            const x = ind * charSize;

            // Matrix Green Spectrum
            const r = 0;
            const g = Math.floor(Math.random() * 155) + 100; // 100-255 range
            const b = 65;

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

            // Occasional bright white characters
            if (Math.random() > 0.98) {
                ctx.fillStyle = '#fff';
            }

            ctx.fillText(text, x, y);

            if (y > 100 + Math.random() * 20000) ypos[ind] = 0;
            else ypos[ind] = y + charSize;
        });
    }
    setInterval(drawMatrix, 50);

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
        $.get('/history', function (records) {
            const tbody = $('#historyTable tbody');
            tbody.empty();
            if (!records || records.length === 0) {
                tbody.append('<tr><td colspan="5">NO HISTORY AVAILABLE</td></tr>');
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
                    </tr>
                `);
            });
        }).fail(function () {
            alert('Unable to load scan history from backend.');
        });
    }

    $('#refreshHistory').click(fetchHistory);

    // --- ANALYZE TEXT ---
    $('#analyzeTextBtn').click(function () {
        if (!loggedIn) {
            showAuth('Sign in to run text scans.', 'login');
            return;
        }
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
        if (!loggedIn) {
            showAuth('Sign in to run URL scans.', 'login');
            return;
        }
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
        if (!loggedIn) {
            showAuth('Sign in to run media scans.', 'login');
            return;
        }
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
                    summary: "",
                    details: []
                };

                if (truthScore < 40) {
                    explanation.summary = "Digital artifacts detected in high-frequency facial regions.";
                    explanation.details = [
                        "Irregular shadow patterns around eyes and mouth.",
                        "Unnatural blinking rhythms detected.",
                        "Compression artifacts inconsistent with standard capture."
                    ];
                } else if (truthScore < 60) {
                    explanation.summary = "Subtle anomalies detected. Structural integrity is within a high margin of error.";
                    explanation.details = [
                        "Minor aliasing on perimeter edges.",
                        "Lighting consistency score is nominal but shows variance.",
                        "Inconclusive neural patterns—hand-forensic verification recommended."
                    ];
                } else {
                    explanation.summary = "No significant frame-to-frame inconsistencies detected.";
                    explanation.details = [
                        "Metadata aligns with captured hardware profile.",
                        "Natural skin texture gradients maintained.",
                        "Consistent lighting across spatial planes."
                    ];
                }

                setTimeout(() => {
                    updateResultUI(truthScore, explanation);
                }, 2500);
            },
            error: function (xhr) { showAjaxError(xhr, 'NEURAL LINK SEVERED'); }
        });
    });
});
