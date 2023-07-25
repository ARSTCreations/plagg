### How Does it Work?
1. Getting an input from the user
2. Detects language used
3. Search google for top 20 most relevant articles
4. Scrape url's webpage content for comparison
5. Calculate Features using Term Frequency-Inverse Document Frequency Vectorization
6. Calculate Cosine Similarity of two feature vectors
7. Returning similarity matrix for each url loop
8. Calculate for max, average, and min value for the plagiarism rate

### Work in Progress
You can try it tho..., but you'd need API Key for Google CSE and the CSE ID itself<br>
Try my CSE! it's filtered to only search for academic domains only<br>
For the API key, you can make it yourself via the google cloud console!
```
CSE_SID = "2214f184193cd4d49"
API_KEY = ""
```

### Todo List
- [x] Creating Concept Program
- [x] Deciding Better Calculation and Search Method
- [x] Creating Base Program
- [x] Implementing Flask API
- [x] Searching for Better Calculation Method
- [ ] Searching for Better Search Method
- [ ] Implementing Academic Literature Classification Option
