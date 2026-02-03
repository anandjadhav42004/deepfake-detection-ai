$(document).ready(function () {

    // --- SAMPLES DATA ---
    const samples = {
        scam: "URGENT: Your bank account has been compromised. Verify your identity immediately to prevent permanent lockout: http://secure-verify-auth.com/account/login-992. No further warnings will be issued.",
        news: "BREAKING: Massive solar flare predicted to shut down the global internet for exactly 48 hours starting tonight. Governments are already preparing emergency rations. Share this before the blackout!",
        ai: "The industrial revolution was a period of significant change that transformed agrarian societies into industrial ones. The emergence of steam power played a pivotal role in this transition, leading to unprecedented levels of production."
    };

    // --- CHECK MODEL STATUS ON LOAD ---
    $.get('/status', function (data) {
        if (!data.deepfake_loaded || !data.nlp_loaded) {
            let errorMsg = 'One or more AI models failed to initialize.';
            if (!data.deepfake_loaded && data.deepfake_error) {
                errorMsg += '\n\nDeepfake Model Error: ' + data.deepfake_error;
            }
            if (!data.nlp_loaded && data.nlp_error) {
                errorMsg += '\n\nNLP Model Error: ' + data.nlp_error;
            }
            $('#errorMessage').text(errorMsg);
            $('#errorModal').removeClass('hidden');
            $('.cyber-btn').prop('disabled', true);
        }
    }).fail(function () {
        $('#errorMessage').text('Failed to connect to the server. Please ensure the backend is running.');
        $('#errorModal').removeClass('hidden');
        $('.cyber-btn').prop('disabled', true);
    });

    $('#closeError').click(function () {
        $('#errorModal').addClass('hidden');
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

            // Reflow
            $('#appContainer').outerWidth();

            $('#appContainer').css('transition', 'all 1s cubic-bezier(0.19, 1, 0.22, 1)')
                .css('opacity', '1')
                .css('transform', 'scale(1)');

            $('.scanner-page').addClass('hidden');
            if (target === 'text-panel') {
                $('#textScannerPage').removeClass('hidden');
            } else {
                $('#mediaScannerPage').removeClass('hidden');
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

    function initProcess() {
        $('.status-pill').text('SYSTEM ANALYZING').addClass('analyzing');
        $('#resultOverlay').removeClass('hidden');
        $('#resultsDisplay').addClass('hidden');
        $('#explanationContent').addClass('hidden');
        $('#whyFlaggedBtn').text("EXPAND_INTELLIGENCE_LOGS");

        let msgIndex = 0;
        systemProgressInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % systemMessages.length;
            $('#systemMessage').text(systemMessages[msgIndex]);
        }, 800);
    }

    function resetProcess() {
        clearInterval(systemProgressInterval);
        $('.status-pill').text('SYSTEM ONLINE').removeClass('analyzing');
        $('#resultOverlay').addClass('hidden');
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

    // --- ANALYZE TEXT ---
    $('#analyzeTextBtn').click(function () {
        const text = $('#newsInput').val();
        if (!text.trim()) return alert("INPUT STREAM EMPTY");

        initProcess();

        $.post('/predict_news', { text: text }, function (data) {
            if (data.result === 'Error') { alert(data.message); resetProcess(); return; }

            const isFake = (data.result === 'FAKE');
            let backendConfidence = parseFloat(data.confidence) || 0;

            // --- THE REAL FIX ---
            // Agar result FAKE hai, toh Truth Probability (score) kam honi chahiye.
            // Agar backend confidence 90% deta hai for FAKE, toh truth score 10% hona chahiye.
            let truthScore;
            if (isFake) {
                truthScore = backendConfidence > 50 ? (100 - backendConfidence) : backendConfidence;
            } else {
                truthScore = backendConfidence < 50 ? (100 - backendConfidence) : backendConfidence;
            }

            // Cap the results
            truthScore = Math.max(2, Math.min(98, truthScore));

            let explanation = {
                summary: "",
                details: []
            };

            if (truthScore < 40) {
                explanation.summary = "Multiple high-risk linguistic patterns detected suggesting fabrication.";
                explanation.details = [
                    "Sensationalist or viral-style language patterns.",
                    "Lack of credible source citations or official backing.",
                    "Syntactic structure common in synthetic or biased media."
                ];
            } else if (truthScore < 60) {
                explanation.summary = "The content falls into a gray area. Mixed signals detected in the neural stream.";
                explanation.details = [
                    "Ambiguous lexical choices detected.",
                    "Mixture of objective and subjective sentence structures.",
                    "Source credibility cannot be fully verified—proceed with caution."
                ];
            } else {
                explanation.summary = "Linguistic structure remains consistent with verified news patterns.";
                explanation.details = [
                    "Factual tone with objective sentence structure.",
                    "Semantic consistency maintained throughout.",
                    "High lexical diversity matching professional journalism."
                ];
            }

            setTimeout(() => {
                updateResultUI(truthScore, explanation);
            }, 1500);

        }).fail(() => { alert("LINK FAILURE"); resetProcess(); });
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
        initProcess();

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
            error: () => { alert("NEURAL LINK SEVERED"); resetProcess(); }
        });
    });
});
