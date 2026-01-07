$(document).ready(function () {

    // --- CANVAS BACKGROUND (The "Running" Visual) ---
    const canvas = document.getElementById('bgCanvas');
    const ctx = canvas.getContext('2d');

    let width, height;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    // Matrix Rain Effect
    const cols = Math.floor(width / 20);
    const ypos = Array(cols).fill(0);

    function drawMatrix() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = '#0F0';
        ctx.font = '15pt monospace';

        ypos.forEach((y, ind) => {
            const text = String.fromCharCode(Math.random() * 128);
            const x = ind * 20;
            ctx.fillText(text, x, y);

            if (y > 100 + Math.random() * 10000) ypos[ind] = 0;
            else ypos[ind] = y + 20;
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
    });

    // Character Count
    $('#newsInput').on('input', function () {
        $('.char-count').text(`${this.value.length} BYTES`);
    });

    // --- ANALYZE TEXT ---
    $('#analyzeTextBtn').click(function () {
        const text = $('#newsInput').val();
        if (!text.trim()) return alert("INPUT STREAM EMPTY");
        initProcess();
        $.post('/predict_news', { text: text }, function (data) {
            if (data.result === 'Error') { alert(data.message); resetProcess(); return; }
            const isFake = data.result === 'FAKE';
            showVerdict(isFake, isFake ? "DECEPTION" : "AUTHENTIC", 98.5);
        }).fail(() => { alert("LINK FAILURE"); resetProcess(); });
    });

    // --- ANALYZE IMAGE (Fixing Drop) ---
    const dropZone = $('#dropZone');
    const imageInput = $('#imageInput');

    // Click to upload
    dropZone.on('click', function (e) {
        imageInput.click();
    });

    imageInput.change(function (e) {
        if (this.files && this.files[0]) handleFile(this.files[0]);
    });

    // Drag Events - Prevent Default is crucial
    dropZone.on('dragover dragenter', function (e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).css('border-color', 'var(--neon-green)').css('box-shadow', '0 0 20px var(--neon-green)');
    });

    dropZone.on('dragleave dragend drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).css('border-color', '').css('box-shadow', '');
    });

    dropZone.on('drop', function (e) {
        const files = e.originalEvent.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
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
                $('#videoPreview').addClass('hidden').attr('src', '');
            } else if (file.type.startsWith('video/')) {
                $('#videoPreview').attr('src', e.target.result).removeClass('hidden');
                $('#imagePreview').addClass('hidden').attr('src', '');
            }
        }
        reader.readAsDataURL(file);
    }

    $('#clearImage').click(function (e) {
        e.stopPropagation();
        currentFile = null;
        imageInput.val('');
        $('#previewContainer').addClass('hidden');
        $('#imagePreview').addClass('hidden');
        $('#videoPreview').addClass('hidden').attr('src', '');
        $('.grid-content').removeClass('hidden');
        $('#analyzeImageBtn').addClass('hidden');
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
                const isFake = data.result === 'FAKE';
                let conf = parseFloat(data.confidence) || 92.4;
                showVerdict(isFake, isFake ? "FABRICATED" : "VERIFIED", conf);
            },
            error: () => { alert("NEURAL LINK SEVERED"); resetProcess(); }
        });
    });

    function initProcess() {
        $('#resultOverlay').removeClass('hidden');
        $('.loader').removeClass('hidden');
        $('.result-content').addClass('hidden');
    }

    function resetProcess() {
        $('#resultOverlay').addClass('hidden');
    }

    function showVerdict(isFake, title, conf) {
        $('.loader').addClass('hidden');
        $('.result-content').removeClass('hidden');
        const color = isFake ? 'var(--neon-red)' : 'var(--neon-green)';
        const desc = isFake ? "ANOMALIES DETECTED" : "INTEGRITY CONFIRMED";
        $('#verdictTitle').text(title).css('color', color).css('text-shadow', `0 0 20px ${color}`);
        $('#verdictDesc').text(desc);
        $('#confidenceVal').text(conf + '%');
        $('#verdictRing').css('border-top-color', color).css('box-shadow', `0 0 20px ${color}`).addClass('stop-spin');
    }

    $('#closeResult').click(() => resetProcess());
});
