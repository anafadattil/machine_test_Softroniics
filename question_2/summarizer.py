from transformers import pipeline

# Load a pre-trained summarization model (BART by default)
from transformers import pipeline

# Use a smaller, safer model
summarizer = pipeline("summarization", model="t5-small")

def summarize_article(article_text, max_length=130, min_length=30):
    summary = summarizer(
        article_text,
        max_length=max_length,  # Max summary length
        min_length=min_length,  # Min summary length
        do_sample=False,        # Disable randomness (for factual accuracy)
    )
    return summary[0]['summary_text']
news_article = """
NATO navies are putting on a display of maritime might in the Baltic Sea this month, as thousands of personnel from 17 countries aboard 50 vessels take part in war games led by the U.S. Navy's 6th Fleet. 

Of the nine countries that share a Baltic Sea coastline, only Russia is not a NATO member, and June's BALTOPS exercise aims to ensure those other countries can work together to defend the area, at a time when Moscow is turning up the heat. 

"This year’s BALTOPS is more than just an exercise," said U.S. Vice Admiral J.T. Anderson in a press release this week. "It’s a visible demonstration of our Alliance’s resolve, adaptability and maritime strength." 

Over the last year there's been growing disquiet about Russia's malign influence in the Baltic Sea region, with several incidents of severed undersea cables. Suspicion has fallen on Russia's fleet of so-called "ghost" or "shadow" ships: hundreds of aging vessels, mostly oil tankers flying under foreign flags that are used to circumvent Western sanctions or trade in military hardware. 
"""

summary = summarize_article(news_article)
print("Summary:", summary)