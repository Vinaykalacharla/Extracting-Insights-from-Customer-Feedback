# Performance Improvement: Persistent Caching of Aspect Sentiment Analysis

## Tasks
- [ ] Create migration script for review_aspect_sentiments cache table
- [ ] Update nlp_utils.py to integrate persistent cache (check DB first, save to DB)
- [ ] Modify admin_api_aspect_sentiment_distribution to use cached results
- [ ] Add cache update logic in upload_review and delete_review endpoints
- [ ] Run migration to create cache table
- [ ] Test performance and correctness

## Details
- New table: review_aspect_sentiments (review_id, aspect_sentiments JSON, cached_at TIMESTAMP)
- Modify analyze_review_detailed to use DB cache
- Aggregate from cache in API endpoint instead of analyzing each review
- Update cache on review add/update/delete

# Admin.html Horizontal Menu Update

- [x] Update interninfos/TODO.md with task steps (current)
- [x] Edit admin.html: Remove sidebar, add horizontal nav, update styles and JS
- [x] Verify edit success and update TODO.md
- [x] Test layout in browser (launch app, navigate to /admin, check tabs and responsiveness)
- [x] Mark task complete if no issues
