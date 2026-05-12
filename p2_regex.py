# %%
import re

# %%
corpus=""" Hellog this is sid
contact on sid@kmv.edu.in or siddk@gmail.com
contact number  +919812311111 or 0987111110.
website is https://nlpxyz.edu/resources.
Popular tags: #NLP #AI #student.

Submission deadline is 25-01-2026. End,
"""

# %%
username=re.findall(r'([\w\.-]+)@\w+\.\w+', corpus)

# %%


# %%
hashtag=re.findall(r"#\w+",corpus)

# %%
dates=re.findall(r"\d{2}-\d{2}-\d{4}",corpus)

# %%
phone_num=re.findall(r'\+?\d{10,13}', corpus)

# %%
print("Usernames:", username)
print("Hashtags:", hashtag)
print("Dates:", dates)
print("Phone Numbers:", phone_num)

# %%


# Usernames: ['sid', 'siddk']
# Hashtags: ['#NLP', '#AI', '#student']
# Dates: ['25-01-2026']
# Phone Numbers: ['+919812311111', '0987111110']
