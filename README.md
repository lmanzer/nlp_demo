# NLP Demo
A demo project for an NLP Application for Financial Services. Scrapes articles from the CNBC website, summarizes the content of the articles, and correlates the content with an Investment Advisor's book of business. 

## Steps to Run:
1. Install virtualenv: `pip install virtualenv`
2. Create your virtual environment: `virtualenv env`
3. Activate your virtual environment: `source env\bin\activate`
4. Install pip packages: `pip install -r requirements.txt`
5. Downloaded nltk data files: `python -m nltk.downloader all`
6. Download Spacy models: `python -m spacy download en_core_web_lg` and `python -m spacy download spacy download en_vectors_web_lg`
7. Create virtual environment kernal for jupyter notebook `ipython kernel install --user --name=nlp_demo`
8. Run jupyter notebook: `jupyter notebook`
 
## Files:

**NLP_Demo.ipynd** Investigates NLP proccess
**Article_Recommendation.ipynd** Runs demo of article recommendations


