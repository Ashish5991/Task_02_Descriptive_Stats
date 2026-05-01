**Project Description and instructions to run all scripts**
In this project, I analyzed a real-world dataset of Facebook political advertisements from the 2024 U.S. Presidential election. The objective was to compute descriptive statistics at both the dataset level and grouped levels using three different approaches:

- Pure Python (standard library)
- Pandas
- Polars

The goal was not just to generate statistics, but to ensure that all three methods produce consistent results while handling real-world data challenges such as missing values, inconsistent formats, and mixed data types.

How to Run the Scripts
Make sure the dataset file is placed in the project folder:
fb_ads_president_scored_anon.csv

Install dependencies:
pip install -r requirements.txt

Run scripts:
python pure_python_grouped_stats.py --file fb_ads_president_scored_anon.csv
python pandas_grouped_stats.py --file fb_ads_president_scored_anon.csv
python polars_grouped_stats.py --file fb_ads_president_scored_anon.csv

**Summary of Findings and Insights**
The dataset contains 246,745 rows and 41 columns, representing a large-scale collection of Facebook political advertisements during the 2024 U.S. Presidential election.
Ad activity is highly concentrated among a few entities. One page alone contributed over 55,000 ads, which is a significant share of the total dataset, indicating that a small number of organizations dominate political advertising on the platform.
From the bylines column, the most frequent advertiser is:
- “HARRIS FOR PRESIDENT” (~49,788 ads)
- Followed by:
 - HARRIS VICTORY FUND
 - BIDEN VICTORY FUND
 - DONALD J. TRUMP FOR PRESIDENT 2024
This shows that a few major political campaigns are responsible for a large portion of ad activity.

Spending & Reach Patterns
The average ad spend is around $1,061, but the median is only $49, which shows:
- Most ads are low-budget
- A few very expensive ads are skewing the average
The maximum spend (~$474,999) confirms the presence of extreme outliers (high-budget campaigns).
Similarly:
Average impressions ~45,601
Median impressions ~3,499

**Comparison of the Three Approaches**


This again shows a highly skewed distribution, where a small number of ads receive very large reach.

**Comparison of the three approaches**
Pure Python
- Required manual implementation of:
    - Mean, median, std deviation
    - Frequency counts
    - Type inference
- Strength:
    - Full control over logic
    - Better understanding of how statistics actually work
- Limitation:
    - More verbose and slower
    - Requires handling edge cases manually

Pandas
- Simplified entire workflow using:
    - .describe()
    - .value_counts()
    - vectorized operations
- Strength:
    - Fast and efficient
    - Handles missing values automatically
    - Industry-standard tool
- Limitation:
    - Abstracts many operations
    - Less visibility into underlying calculations

Polars
- Similar to Pandas but:
    - More performance-oriented
    - Uses lazy execution and optimized memory handling
- Strength:
    - Faster for large datasets
    - Efficient processing
- Limitation:
    - Slightly stricter behavior
    - Requires more explicit handling in some cases
