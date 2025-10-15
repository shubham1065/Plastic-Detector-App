// The URL of your live Render API endpoint
// This must be set to your actual live Render URL, e.g., 'https://plastic-detector-api.onrender.com/predict'
const API_URL = 'https://plastic-detector-api.onrender.com/predict';

// --- PLASTIC DATA REFERENCE (Used for front-end display) ---
const PLASTIC_DATA = {
    'PET': {
        name: 'Polyethylene Terephthalate',
        uses: 'Water/soda bottles, clear food packaging, polyester fiber.',
        notes: 'Widely recycled (RIC #1). Often the most widely accepted plastic.',
        color: '#007bff'
    },
    'PE': {
        name: 'Polyethylene (HDPE/LDPE)',
        uses: 'Milk jugs, detergent bottles, plastic bags, rigid containers.',
        notes: 'Commonly recycled (RIC #2 or #4). High-Density (HDPE) is most valuable.',
        color: '#28a745'
    },
    'PP': {
        name: 'Polypropylene',
        uses: 'Yogurt tubs, bottle caps, medicine bottles, food containers.',
        notes: 'Recyclability varies by location (RIC #5). Increasingly accepted.',
        color: '#ffc107'
    },
    'PS': {
        name: 'Polystyrene',
        uses: 'Foam cups, disposable cutlery, CD cases, foam packaging.',
        notes: 'Rarely recycled curbside (RIC #6). Often landfilled.',
        color: '#dc3545'
    },
    'PC': {
        name: 'Polycarbonate',
        uses: 'CDs/DVDs, large water cooler bottles, lenses.',
        notes: 'Included in "Others" (RIC #7). Not typically recycled due to complex composition.',
        color: '#6f42c1'
    },
    'others': {
        name: 'Miscellaneous / Composite Plastics',
        uses: 'Mixed polymers, multi-layer packaging, non-standard resins.',
        notes: 'Generally not recyclable via standard programs (RIC #7).',
        color: '#6c757d'
    }
};

// --- DOM ELEMENT REFERENCES ---
const imageInput = document.getElementById('imageInput');
const uploadedImage = document.getElementById('uploadedImage');
const predictButton = document.getElementById('predictButton');
const resultsBox = document.getElementById('results');
const loadingIndicator = document.getElementById('loading');
const predictedClassElement = document.getElementById('predictedClass');
const confidenceElement = document.getElementById('confidence');
const statusMessageElement = document.getElementById('statusMessage');
const polymerDetailsDiv = document.getElementById('polymerDetails');
const referenceSection = document.getElementById('referenceSection');
const darkModeToggle = document.getElementById('darkModeToggle');


// --- DARK MODE LOGIC ---

// Set the theme when the page loads
function loadThemePreference() {
    const isDarkMode = localStorage.getItem('darkMode') === 'enabled';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeToggle.textContent = 'Disable Dark Mode';
    } else {
        darkModeToggle.textContent = 'Enable Dark Mode';
    }
}

// Toggle the theme and save preference
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDarkMode = document.body.classList.contains('dark-mode');

    if (isDarkMode) {
        localStorage.setItem('darkMode', 'enabled');
        darkModeToggle.textContent = 'Disable Dark Mode';
    } else {
        localStorage.setItem('darkMode', 'disabled');
        darkModeToggle.textContent = 'Enable Dark Mode';
    }
}

// Attach the theme toggler listener
darkModeToggle.addEventListener('click', toggleTheme);
document.addEventListener('DOMContentLoaded', loadThemePreference);


// --- UI and FILE HANDLER ---

imageInput.addEventListener('change', function(event) {
    const file = event.target.files[0];

    // Reset results display
    resultsBox.style.display = 'none';
    statusMessageElement.textContent = 'Waiting for prediction...';
    polymerDetailsDiv.innerHTML = '';
    referenceSection.style.display = 'none'; // Ensure table is hidden on new upload

    if (file) {
        // Display the uploaded image
        uploadedImage.src = URL.createObjectURL(file);
        uploadedImage.style.display = 'block';
        predictButton.disabled = false; // Enable the button
    } else {
        uploadedImage.style.display = 'none';
        predictButton.disabled = true; 
    }
});


// --- PREDICTION LOGIC ---

predictButton.addEventListener('click', async () => {
    const file = imageInput.files[0];

    if (!file) return;
    
    // UI Feedback: Show loading indicator and disable button
    resultsBox.style.display = 'block';
    loadingIndicator.style.display = 'block';
    predictButton.disabled = true;
    statusMessageElement.textContent = 'Processing...';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        // --- FINAL CACHE-BYPASS FIX ---
        // 1. Define the unique URL (Appends timestamp to bypass browser/proxy caching)
        const uniqueApiUrl = `${API_URL}?t=${Date.now()}`;

        // 2. Use the unique URL in the fetch call
        const response = await fetch(uniqueApiUrl, {
            method: 'POST',
            body: formData,
        });
        // --- END OF CACHE-BYPASS FIX ---

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Get data and details
            const predictedCode = data.predicted_class;
            const details = PLASTIC_DATA[predictedCode] || PLASTIC_DATA['others']; // Fallback
            
            // 1. Update main prediction elements
            statusMessageElement.textContent = 'Prediction Complete';
            predictedClassElement.textContent = `${predictedCode} (${details.name})`;
            confidenceElement.textContent = data.confidence;
            
            // 2. Inject specific polymer details
            polymerDetailsDiv.innerHTML = `
                <p><strong>Code Name:</strong> ${details.name}</p>
                <p><strong>Common Uses:</strong> ${details.uses}</p>
                <p><strong>Recycling Status:</strong> ${details.notes}</p>
            `;
            polymerDetailsDiv.style.display = 'block'; // Make sure details are visible

        } else {
            // Handle API errors (status != 200 or status=error)
            statusMessageElement.textContent = 'Error during prediction.';
            predictedClassElement.textContent = data.error || 'Server processing failed.';
            confidenceElement.textContent = 'Check server logs.';
            polymerDetailsDiv.style.display = 'none';
        }
    } catch (error) {
        // Handle Network/CORS failure
        console.error('Network or Fetch Error:', error);
        statusMessageElement.textContent = 'Network Error';
        predictedClassElement.textContent = 'Could not reach API.';
        confidenceElement.textContent = 'Ensure backend is running.';
        polymerDetailsDiv.style.display = 'none';
    } finally {
        // UI Feedback: Hide loading indicator and re-enable button
        loadingIndicator.style.display = 'none';
        predictButton.disabled = false;
    }
});


// --- TOGGLE TABLE LOGIC ---

document.getElementById('toggleReference').addEventListener('click', function(e) {
    e.preventDefault();
    const isHidden = referenceSection.style.display === 'none' || referenceSection.style.display === '';
    
    if (isHidden) {
        referenceSection.style.display = 'block';
        this.textContent = 'Hide detailed information «';
    } else {
        referenceSection.style.display = 'none';
        this.textContent = 'Know more about all types »';
    }
});
