In this project, I analyzed a real-world dataset of Facebook political advertisements from the 2024 U.S. Presidential election. The goal was to compute descriptive statistics at both the dataset level and grouped levels using three different approaches: pure Python, Pandas, and Polars.

I started by working with Python’s standard library to manually compute statistics such as mean, median, standard deviation, and frequency distributions. This helped me understand how these calculations actually work behind the scenes, especially when dealing with messy real-world data like missing values, inconsistent formats, and mixed data types.

After that, I replicated the same analysis using Pandas and Polars. The main objective was to ensure that all three scripts produce consistent results, even though they use different tools and approaches. This comparison helped me understand how libraries like Pandas simplify many operations that require more effort in pure Python.

**What I Did**
Loaded and explored a real-world dataset of political ads
Identified numeric, categorical, and date-like columns using type inference
Handled missing values and inconsistent formats (e.g., $, commas, empty strings)
Computed descriptive statistics for:
Numeric columns (mean, min, max, std, median)
Categorical columns (unique values, mode, top frequencies)
Performed grouped analysis by key columns to understand patterns within subsets of the data
Ensured that all three implementations (Pure Python, Pandas, Polars) produce consistent outputs
Exported results into structured JSON files for comparison and reproducibility

**Key Learning**
One of the most important parts of this project was understanding the difference between manual computation and library-based analysis.

- In pure Python, I had to explicitly handle edge cases like missing values and type conversion, which gave me a deeper understanding of how statistics are calculated.
- In Pandas and Polars, many of these operations are optimized and abstracted, making the process faster but less transparent.

**Outcome**
This project helped me realize that while libraries are powerful, understanding the underlying logic is critical to trust and validate results—especially when working with real-world data.

By the end of this project, I was able to:

- Build three independent scripts that compute identical statistics
- Perform both dataset-level and grouped-level analysis
- Handle messy and inconsistent real-world data reliably
- Compare different data processing approaches and understand their trade-offs
