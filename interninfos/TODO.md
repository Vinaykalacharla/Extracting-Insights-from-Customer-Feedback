# Profile Review Analysis Enhancement

## Plan Implementation Steps

### 1. Create NLP Utilities (`nlp_utils.py`)
- [x] Extract NLP functions from routes.py
- [x] Add aspect extraction functionality
- [x] Add detailed sentiment analysis for specific aspects
- [x] Add functions to generate analysis summary

### 2. Add New Route (`routes.py`)
- [x] Create `/review_analysis/<review_id>` route
- [x] Return detailed analysis data in JSON format
- [x] Include aspect extraction, sentiment breakdown, and summary

### 3. Modify Profile Template (`profile.html`)
- [x] Remove sentiment badges, confidence scores from main review display
- [x] Add "View" button to each review card
- [x] Create modal/popup structure for detailed analysis
- [x] Add JavaScript for popup functionality and data loading

### 4. Create Analysis Popup/Modal
- [x] Design modal with sections for review text with aspect highlighting
- [x] Add aspect sentiment breakdown table
- [x] Add aspect sentiment scores visualization
- [x] Add analysis summary with statistics

### 5. Add Frontend JavaScript
- [x] Handle popup opening/closing
- [x] Fetch detailed analysis data via AJAX
- [x] Render aspect breakdown and charts
- [x] Handle responsive design

## Testing Checklist
- [ ] Test new detailed analysis functionality
- [ ] Verify popup displays correctly
- [ ] Test aspect extraction accuracy
- [ ] Ensure responsive design works
- [ ] Test with various review types
