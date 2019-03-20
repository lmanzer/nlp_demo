import urllib 
from bs4 import BeautifulSoup


def extract_cnbc_article_info(url):
    
    page_html = urllib.request.urlopen(url)
    souped_html = BeautifulSoup(page_html, 'html.parser')
    
    body_html = souped_html.find('body')
    content_html = body_html.find('div', attrs={'id': 'cnbc-contents'})
    content_body_html = content_html.find('div', attrs={'class': 'cnbc-body'})
    story_top_html = content_body_html.find('div', attrs={'class': 'story-top'})

    # Extract Article Title
    title = story_top_html.find('h1', attrs={'class': 'title'}).text

    # Extract Article Summary
    summary_html = story_top_html.find('div', attrs={'id': 'article_deck'}) 
    summary_list_html = summary_html.findAll('li')
    summary_list = []
    for summary in summary_list_html:
        _summary_text = summary.text
        summary_list.append(_summary_text)
    
    # Extract Reporter Name
    section_html = content_body_html.find('section', attrs={'class': 'cols2'})
    reporter_html = section_html.find('div', attrs={'class': 'social-reporter'})
    reporter_name = reporter_html.find('span', attrs={'class': 'name'}).text

    # Extract Article Text
    article_html = section_html.find('div', attrs={'id': 'article_body'})
    group_container_list = article_html.findAll('div', attrs={'class': 'group-container'})
    article_text = ""
    for group in group_container_list:
        _article_text = group.find('div', attrs={'class': 'group'})
        article_text += _article_text.text

    return {
        'title': title,
        'reporter': reporter_name,
        'summary': summary_list,
        'article': article_text,
        'url': url,
    }


def extract_cbc_article_info(url):
   
    page_html = urllib.request.urlopen(url)
    souped_html = BeautifulSoup(page_html, 'html.parser')
    
    body_html = souped_html.find('body')
    content_html = body_html.find('main', attrs={'id': 'content'})
    
    # Extract Article Title
    title = content_html.find('h1', attrs={'class': 'detailHeadline'}).text
    
    # Extract Reporter Name
    reporter_name = content_html.find('span', attrs={'class': 'authorText'})
    if reporter_name is None:
        byline_text =  content_html.find('div', attrs={'class': 'bylineDetails'}).text
        reporter_name = byline_text.split(' · ')[0]
        print(reporter_name)

    # Extract Article Summary
    summary_html = content_html.find('h2', attrs={'class': 'deck'}) 

    # Extract Article Text
    article_html = content_html.find('div', attrs={'class': 'story'})
    group_container_list = article_html.findAll('p')
    article_text = ""
    for text in group_container_list:
        article_text += f' {text.text} .'

    return {
        'title': title,
        'reporter': reporter_name,
        'summary': summary_html.text,
        'article': article_text,
        'url': url,
    }
